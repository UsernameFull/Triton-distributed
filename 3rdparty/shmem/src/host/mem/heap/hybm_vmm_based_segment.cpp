/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <bitset>

#include "acl/acl.h"
#include "acl/acl_rt.h"

#include "dl_comm_def.h"
#include "dl_api.h"
#include "dl_hal_api.h"
#include "dl_acl_api.h"
#include "shmemi_num_util.h"
#include "shmemi_logger.h"
#include "devmm_svm_gva.h"
#include "mem_entity_def.h"
#include "shmemi_net_util.h"
#include "hybm_ex_info_transfer.h"
#include "hybm_mem_common.h"
#include "hybm_vmm_based_segment.h"

using namespace shm;

Result HybmVmmBasedSegment::ValidateOptions() noexcept
{
    if (options_.size == 0 || (options_.size % DEVICE_LARGE_PAGE_SIZE) != 0) {
        SHM_LOG_ERROR("Invalid options segType:" << options_.segType << " size:" << options_.size);
        return ACLSHMEM_INVALID_PARAM;
    }

    if (UINT64_MAX / options_.size < options_.rankCnt) {
        SHM_LOG_ERROR("Validate options error rankCnt(" << options_.rankCnt << ") size(" << options_.size);
        return ACLSHMEM_INVALID_PARAM;
    }

    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::ReserveMemorySpace(void **address) noexcept
{
    SHM_ASSERT_RETURN(address != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_LOG_AND_RETURN(ValidateOptions() == ACLSHMEM_SUCCESS, "Failed to validate options.", ACLSHMEM_INVALID_PARAM);
    if (globalVirtualAddress_ != nullptr) {
        SHM_LOG_ERROR("already prepare virtual memory.");
        return ACLSHMEM_INNER_ERROR;
    }

    if (options_.rankId >= options_.rankCnt) {
        SHM_LOG_ERROR("rank(" << options_.rankId << ") but total " << options_.rankCnt);
        return ACLSHMEM_INVALID_PARAM;
    }

    void *base = nullptr;
    uint64_t expectSt = HYBM_GVM_START_ADDR;
    size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
    size_t totalReservedSize = options_.rankCnt * reserveAlignedSize;
    uint64_t flag = MEM_RSV_TYPE_REMOTE_MAP;
    flag |= options_.shared ? MEM_RSV_TYPE_DEVICE_SHARE : 0U;
    auto ret = DlHalApi::HalMemAddressReserve(&base, totalReservedSize, 0, reinterpret_cast<void *>(expectSt), flag);
    if (ret != 0 || base == nullptr) {
        SHM_LOG_ERROR("prepare virtual memory total size(" << totalReservedSize << ") failed. ret: " << ret);
        return ACLSHMEM_MALLOC_FAILED;
    }

    SHM_LOG_INFO("success to reserve total memory space for logic deviceid " << logicDeviceId_ 
                << ", vaddr: " << base << " total size: " << totalReservedSize);

    uint8_t* currentAddr = static_cast<uint8_t*>(base);
    for (uint32_t i = 0; i < options_.rankCnt; i++) {
        reservedVirtualAddresses_.emplace_back(reinterpret_cast<uint64_t>(currentAddr));
        SHM_LOG_INFO("rankId: " << i << ", vaddr: " << (void*)currentAddr << " size: " << options_.size << " align_size: " << reserveAlignedSize);
        currentAddr += reserveAlignedSize;
    }

    globalVirtualAddress_ = reinterpret_cast<uint8_t*>(reservedVirtualAddresses_[0]);
    allocatedSize_ = 0UL;
    sliceCount_ = 0;
    *address = reinterpret_cast<void *>(reservedVirtualAddresses_[0]);
    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::UnReserveMemorySpace() noexcept
{
    while (!slices_.empty()) {
        auto slice = slices_.begin()->second.slice;
        ReleaseSliceMemory(slice);
    }
    allocatedSize_ = 0;
    sliceCount_ = 0;
    if (globalVirtualAddress_ != nullptr) {
        DlHalApi::HalMemAddressFree(reinterpret_cast<void *>(globalVirtualAddress_));
        reservedVirtualAddresses_.clear();
        totalVirtualSize_ = 0;
        globalVirtualAddress_ = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::AllocLocalMemory(uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept
{
    size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
    if ((size % DEVICE_LARGE_PAGE_SIZE) != 0UL || size + allocatedSize_ > reserveAlignedSize) {
        SHM_LOG_ERROR("invalid allocate memory size : " << size << ", now used " << allocatedSize_ << " of "
                                                       << options_.size);
        return ACLSHMEM_INVALID_PARAM;
    }

    SHM_ASSERT_RETURN(globalVirtualAddress_ != nullptr, ACLSHMEM_NOT_INITED);

    auto localVirtualBase = reservedVirtualAddresses_[options_.rankId];
    uint64_t allocAddr = reinterpret_cast<uint64_t>(localVirtualBase + allocatedSize_);
    drv_mem_handle_t *handle = nullptr;
    Result ret = ACLSHMEM_SUCCESS;
    drv_mem_prop prop{};
    prop = {MEM_DEV_SIDE, static_cast<uint32_t>(logicDeviceId_), 0, MEM_NORMAL_PAGE_TYPE, MEM_HBM_TYPE, 0};
    ret = DlHalApi::HalMemCreate(&handle, size, &prop, 0);
    SHM_VALIDATE_RETURN(ret == ACLSHMEM_SUCCESS,
                       "HalMemCreate failed: " << ret << " segType:" << options_.segType << " devId:" << logicDeviceId_
                                               << " size:" << size,
                       ACLSHMEM_DL_FUNC_FAILED);
    allocatedSize_ += size;
    slice = std::make_shared<MemSlice>(sliceCount_++, MEM_TYPE_DEVICE_HBM, MEM_PT_TYPE_GVM, allocAddr, size);
    SHM_ASSERT_RETURN(slice != nullptr, ACLSHMEM_MALLOC_FAILED);
    slices_.emplace(slice->index_, MemSliceStatus(slice, reinterpret_cast<void *>(handle)));

    MemShareHandle sHandle = {};
    ret = ExportInner(slice, sHandle);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("export failed: " << ret);
        DlHalApi::HalMemRelease(handle);
        return ACLSHMEM_INNER_ERROR;
    }
    ret = DlHalApi::HalMemMap(reinterpret_cast<void *>(allocAddr), size, 0, handle, 0);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("HalMemMap failed: " << ret);
        DlHalApi::HalMemRelease(handle);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    SHM_LOG_DEBUG("alloc mem success, addr:0x" << std::hex << allocAddr << " size:0x" << size);
    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::RegisterMemory(const void *addr, uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept
{
    slice = std::make_shared<MemSlice>(sliceCount_++, MEM_TYPE_DEVICE_HBM, MEM_PT_TYPE_SVM, reinterpret_cast<uint64_t>(addr), size);
    SHM_LOG_INFO("HybmVmmBasedSegment: RegisterMemory success, size: " << size << " addr: " << std::hex << addr);
    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::ReleaseSliceMemory(const std::shared_ptr<MemSlice> &slice) noexcept
{
    SHM_VALIDATE_RETURN(slice != nullptr, "input slice is nullptr", ACLSHMEM_INVALID_PARAM);

    auto pos = slices_.find(slice->index_);
    if (pos == slices_.end()) {
        SHM_LOG_ERROR("input slice(idx:" << slice->index_ << ") not exist.");
        return ACLSHMEM_INVALID_PARAM;
    }
    if (pos->second.slice != slice) {
        SHM_LOG_ERROR("input slice(magic:" << std::hex << slice->magic_ << ") not match.");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto res = DlHalApi::HalMemUnmap(reinterpret_cast<void *>(slice->vAddress_));
    SHM_LOG_INFO("unmap slice(idx:" << slice->index_ << "), size: " << slice->size_ << " return:" << res);

    res = DlHalApi::HalMemRelease(reinterpret_cast<drv_mem_handle_t *>(pos->second.handle));
    SHM_LOG_INFO("release slice(idx:" << slice->index_ << "), size: " << slice->size_ << " return:" << res);

    slices_.erase(pos);
    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::Export(std::string &exInfo) noexcept
{
    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::ExportInner(const std::shared_ptr<MemSlice> &slice, MemShareHandle &sHandle) noexcept
{
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip export");
        return ACLSHMEM_SUCCESS;
    }

    MemExportInfo info;
    std::string exInfo;
    auto pos = slices_.find(slice->index_);
    auto ret = DlHalApi::HalMemExport(reinterpret_cast<drv_mem_handle_t *>(pos->second.handle), MEM_HANDLE_TYPE_FABRIC,
                                      0, &info.shareHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("create shm memory key failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    uint64_t shareable = 0U;
    uint32_t sId;
    ret = DlHalApi::HalMemTransShareableHandle(MEM_HANDLE_TYPE_FABRIC, &info.shareHandle, &sId, &shareable);
    SHM_VALIDATE_RETURN(ret == ACLSHMEM_SUCCESS, "HalMemTransShareableHandle failed:" << ret, ACLSHMEM_INNER_ERROR);
    struct ShareHandleAttr attr = {};
    attr.enableFlag = SHR_HANDLE_NO_WLIST_ENABLE;
    ret = DlHalApi::HalMemShareHandleSetAttribute(shareable, SHR_HANDLE_ATTR_NO_WLIST_IN_SERVER, attr);
    SHM_VALIDATE_RETURN(ret == ACLSHMEM_SUCCESS, "HalMemShareHandleSetAttribute failed:" << ret, ACLSHMEM_INNER_ERROR);
    info.deviceId = options_.devId;
    info.logicDeviceId = logicDeviceId_;
    info.magic = HBM_SLICE_EXPORT_INFO_MAGIC;
    info.version = EXPORT_INFO_VERSION;
    auto localVirtualBase = reservedVirtualAddresses_[options_.rankId];
    info.mappingOffset = slice->vAddress_ - (uint64_t)(ptrdiff_t)localVirtualBase;
    info.sliceIndex = static_cast<uint32_t>(slice->index_);
    info.rankId = options_.rankId;
    info.size = slice->size_;
    info.sdid = sdid_;
    info.serverId = serverId_;
    info.superPodId = superPodId_;
    ret = LiteralExInfoTranslater<MemExportInfo>{}.Serialize(info, exInfo);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("export info failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    exportMap_[slice->index_] = exInfo;
    sHandle = info.shareHandle;
    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::Export(const std::shared_ptr<MemSlice> &slice, std::string &exInfo) noexcept
{
    SHM_ASSERT_RETURN(slice != nullptr, ACLSHMEM_INVALID_PARAM);
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip export");
        return ACLSHMEM_SUCCESS;
    }
    auto pos = slices_.find(slice->index_);
    SHM_VALIDATE_RETURN(pos != slices_.end(), "input slice(idx:" << slice->index_ << ") not exist.", ACLSHMEM_INVALID_PARAM);
    SHM_VALIDATE_RETURN(pos->second.slice == slice, "input slice(magic:" << std::hex << slice->magic_ << ") not match.",
                       ACLSHMEM_INVALID_PARAM);

    auto exp = exportMap_.find(slice->index_);
    if (exp != exportMap_.end()) {
        exInfo = exp->second;
        return ACLSHMEM_SUCCESS;
    } else {
        return ACLSHMEM_INNER_ERROR;
    }
}

Result HybmVmmBasedSegment::GetExportSliceSize(size_t &size) noexcept
{
    size = sizeof(MemExportInfo);
    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::Import(const std::vector<std::string> &allExInfo, void *addresses[]) noexcept
{
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip import");
        return ACLSHMEM_SUCCESS;
    }

    std::map<uint16_t, MemExportInfo> importMap;
    LiteralExInfoTranslater<MemExportInfo> translator;
    uint64_t exportMagic = HBM_SLICE_EXPORT_INFO_MAGIC;
    uint64_t heapSize = 0;
    std::vector<MemExportInfo> desInfos{allExInfo.size()};
    for (auto i = 0U; i < allExInfo.size(); i++) {
        auto ret = translator.Deserialize(allExInfo[i], desInfos[i]);
        if (ret != 0) {
            SHM_LOG_ERROR("deserialize imported info(" << i << ") failed.");
            return ACLSHMEM_INVALID_PARAM;
        }

        if (desInfos[i].magic != exportMagic) {
            SHM_LOG_ERROR("import info(" << i << ") magic(" << desInfos[i].magic << ") invalid.");
            return ACLSHMEM_INVALID_PARAM;
        }
        if (i == 0) {
            heapSize = desInfos[i].size;
        } else if (heapSize != desInfos[i].size) {
            SHM_LOG_ERROR("import info(" << i << ") size(" << desInfos[i].size << ") invalid.");
            return ACLSHMEM_INVALID_PARAM;
        }

        if (options_.segType == HYBM_MST_HBM && desInfos[i].rankId != options_.rankId &&
            CanLocalHostReaches(desInfos[i].superPodId, desInfos[i].serverId, desInfos[i].logicDeviceId)) {
            auto ret = DlAclApi::AclrtDeviceEnablePeerAccess(desInfos[i].deviceId, 0);
            if (ret != 0) {
                SHM_LOG_ERROR("enable device access failed:" << ret << " local_device:" << deviceId_
                                                            << " remote_device:" << desInfos[i].deviceId
                                                            << " logic_device:" << logicDeviceId_
                                                            << " remote_logic:" << desInfos[i].logicDeviceId);
                return ACLSHMEM_DL_FUNC_FAILED;
            }
            SHM_LOG_DEBUG("enable device access success local_device:" << deviceId_
                                                                       << " remote_device:" << desInfos[i].deviceId
                                                                       << " logic_device:" << logicDeviceId_
                                                                       << " remote_logic:" << desInfos[i].logicDeviceId);
        }
        importMap.emplace(desInfos[i].rankId, desInfos[i]);
    }
    importMap_ = std::move(importMap);

    try {
        std::copy(desInfos.begin(), desInfos.end(), std::back_inserter(imports_));
    } catch (...) {
        SHM_LOG_ERROR("copy failed.");
        return ACLSHMEM_MALLOC_FAILED;
    }
    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::Mmap() noexcept
{
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip map");
        return ACLSHMEM_SUCCESS;
    }

    if (imports_.empty()) {
        return ACLSHMEM_SUCCESS;
    }
    for (auto &im : imports_) {
        if (im.rankId == options_.rankId) {
            continue;
        }

        auto remoteAddress = reservedVirtualAddresses_[im.rankId];
        if (mappedMem_.find(remoteAddress) != mappedMem_.end()) {
            SHM_LOG_INFO("remote slice on rank(" << im.rankId << ") has maped: " << (void *)remoteAddress);
            continue;
        }

        if (!CanMapRemote(im)) {
            SHM_LOG_INFO("remote slice on rank(" << im.rankId << ") SDMA cannot reaches.");
            continue;
        }

        SHM_LOG_DEBUG("remote slice on rank(" << im.rankId << ") should map to: " << (void *)remoteAddress
                                             << ", size = " << im.size);
        drv_mem_handle_t *handle = nullptr;
        auto ret = DlHalApi::HalMemImport(MEM_HANDLE_TYPE_FABRIC, &im.shareHandle, logicDeviceId_, &handle);
        SHM_VALIDATE_RETURN(
            ret == ACLSHMEM_SUCCESS, "HalMemImport memory failed:" << ret << " local sdid:" << sdid_ << " remote ssid:" << im.sdid,
            ACLSHMEM_INNER_ERROR);

        ret = DlHalApi::HalMemMap(reinterpret_cast<void *>(remoteAddress), im.size, 0, handle, 0);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("HalMemMap memory failed:" << ret);
            DlHalApi::HalMemRelease(handle);
            return ACLSHMEM_INNER_ERROR;
        }
        mappedMem_.emplace(remoteAddress, handle);
    }
    imports_.clear();
    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::Unmap() noexcept
{
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip unmap");
        return ACLSHMEM_SUCCESS;
    }

    for (auto pd : mappedMem_) {
        DlHalApi::HalMemUnmap(reinterpret_cast<void *>(pd.first));
        DlHalApi::HalMemRelease(pd.second);
    }
    mappedMem_.clear();
    return ACLSHMEM_SUCCESS;
}

Result HybmVmmBasedSegment::RemoveImported(const std::vector<uint32_t> &ranks) noexcept
{
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip remove");
        return ACLSHMEM_SUCCESS;
    }
    for (auto &rank : ranks) {
        if (rank >= options_.rankCnt) {
            SHM_LOG_ERROR("input rank is invalid! rank:" << rank << " rankSize:" << options_.rankCnt);
            return ACLSHMEM_INVALID_PARAM;
        }
    }
    size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
    for (auto &rank : ranks) {
        uint64_t addr = reinterpret_cast<uint64_t>(globalVirtualAddress_) + reserveAlignedSize * rank;
        auto it = mappedMem_.lower_bound(addr);
        auto st = it;
        while (it != mappedMem_.end() && (*it).first < addr + reserveAlignedSize) {
            DlHalApi::HalMemUnmap(reinterpret_cast<void *>((*it).first));
            DlHalApi::HalMemRelease((*it).second);
            it++;
        }

        if (st != it) {
            mappedMem_.erase(st, it);
        }
    }
    return ACLSHMEM_SUCCESS;
}

std::shared_ptr<MemSlice> HybmVmmBasedSegment::GetMemSlice(hybm_mem_slice_t slice) const noexcept
{
    auto index = MemSlice::GetIndexFrom(slice);
    auto pos = slices_.find(index);
    if (pos == slices_.end()) {
        return nullptr;
    }

    auto target = pos->second.slice;
    if (!target->ValidateId(slice)) {
        return nullptr;
    }

    return target;
}

bool HybmVmmBasedSegment::MemoryInRange(const void *begin, uint64_t size) const noexcept
{
    if (begin < globalVirtualAddress_) {
        return false;
    }

    if (reinterpret_cast<const uint8_t *>(begin) + size > globalVirtualAddress_ + totalVirtualSize_) {
        return false;
    }

    return true;
}

bool HybmVmmBasedSegment::CanMapRemote(const MemExportInfo &rmi) noexcept
{
    return IsSdmaAccessible(rmi.superPodId, rmi.serverId, rmi.logicDeviceId);
}

void HybmVmmBasedSegment::GetDeviceInfo(uint32_t &sdId, uint32_t &serverId, uint32_t &superPodId) noexcept
{
    sdId = sdid_;
    serverId = serverId_;
    superPodId = superPodId_;
}

bool HybmVmmBasedSegment::GetRankIdByAddr(const void *addr, uint64_t size, uint32_t &rankId) const noexcept
{
    return true;
}

bool HybmVmmBasedSegment::CheckSdmaReaches(uint32_t remoteRankId) const noexcept
{
    auto pos = importMap_.find(static_cast<uint16_t>(remoteRankId));
    if (pos == importMap_.end()) {
        return false;
    }

    return IsSdmaAccessible(pos->second.superPodId, pos->second.serverId, pos->second.logicDeviceId);
}
