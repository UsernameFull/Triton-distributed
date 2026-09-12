/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE

#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "acl/acl.h"
#include "acl/acl_rt.h"

#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/dev.h"
#include "runtime/rt_ffts.h"

#include "dl_comm_def.h"
#include "mem_entity_def.h"
#include "devmm_svm_gva.h"
#include "shmemi_logger.h"
#include "shmemi_net_util.h"
#include "hybm_ex_info_transfer.h"
#include "hybm_device_mem_segment.h"

namespace shm {
Result MemSegmentDevice::ValidateOptions() noexcept
{
    if (options_.size == 0 || options_.devId < 0 ||
        (options_.size % DEVICE_LARGE_PAGE_SIZE) != 0) {
        return ACLSHMEM_INVALID_PARAM;
    }

    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::ReserveEachPeMemorySpace(size_t reserveAlignedSize, size_t totalReservedSize, uint64_t expectSt) noexcept
{
    // because 950 not support specify va
    void *base = nullptr;
    if (socType_ == AscendSocType::ASCEND_950) {
        auto ret = aclrtReserveMemAddress(&base, totalReservedSize, 0, nullptr, 1);
        if (ret != 0 || base == 0) {
            SHM_LOG_ERROR("prepare virtual memory size(" << totalReservedSize << ") failed. ret: " << ret);
            return ACLSHMEM_MALLOC_FAILED;
        }
        uint8_t* currentAddr = static_cast<uint8_t*>(base);
        for (uint32_t i = 0; i < options_.rankCnt; i++) {
            reservedVirtualAddresses_.emplace_back(reinterpret_cast<uint64_t>(currentAddr));
            SHM_LOG_INFO("rankId: " << i << ", vaddr: " << (void*)currentAddr << " size: " << options_.size << " align_size: " << reserveAlignedSize);
            currentAddr += reserveAlignedSize;
        }
        return ACLSHMEM_SUCCESS;
    }
    uint8_t *curBase = (uint8_t *)expectSt;
    for (uint32_t i = 0; i < options_.rankCnt; i++) {
        auto ret = aclrtReserveMemAddress(&base, reserveAlignedSize, 0, curBase, 1);
        if (ret != 0 || base == 0) {
            SHM_LOG_ERROR("prepare virtual memory size(" << totalVirtualSize_ << ") failed. ret: " << ret);
            return ACLSHMEM_MALLOC_FAILED;
        }
        SHM_LOG_INFO("success to reserve memory space for logic deviceid " << logicDeviceId_ << ", vaddr: " << (void *)base << " size: " << options_.size << ", rankId: " << i);
        totalVirtualSize_ += reserveAlignedSize;
        reservedVirtualAddresses_.emplace_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)));
        curBase = (uint8_t *)base + reserveAlignedSize;
    }
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::ReserveMemorySpace(void **address) noexcept
{
    if (globalVirtualAddress_ != nullptr) {
        SHM_LOG_ERROR("already prepare virtual memory.");
        return ACLSHMEM_INNER_ERROR;
    }
    size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
    size_t totalReservedSize = options_.rankCnt * reserveAlignedSize;
    uint64_t expectSt;
    if (FindAvaliableVirtualAddr(totalReservedSize, expectSt) != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("prepare virtual memory size(" << totalReservedSize << ") failed.");
        return ACLSHMEM_MALLOC_FAILED;
    }
    if (ReserveEachPeMemorySpace(reserveAlignedSize, totalReservedSize, expectSt) != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("ReserveEachPeMemorySpace virtual memory size(" << totalReservedSize << ") failed.");
        return ACLSHMEM_MALLOC_FAILED;
    }

    globalVirtualAddress_ = reinterpret_cast<uint8_t *>(reservedVirtualAddresses_[0]);
    allocatedSize_ = 0UL;
    sliceCount_ = 0;
    *address = reinterpret_cast<void *>(reservedVirtualAddresses_[0]);
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::FindAvaliableVirtualAddr(uint64_t size, uint64_t &baseVa) noexcept
{
    void *base;
    auto ret = aclrtReserveMemAddress(&base, size, 0, nullptr, 1);
    if (ret != 0 || base == 0) {
        SHM_LOG_ERROR("prepare virtual memory size to (" << size << ") failed. ret: " << ret);
        return ACLSHMEM_MALLOC_FAILED;
    }
    baseVa = (uint64_t)base;
    ret = aclrtReleaseMemAddress(base);
    if (ret != 0) {
        SHM_LOG_ERROR("aclrtReleaseMemAddress failed: " << ret);
    }
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::SetMemAccess() noexcept
{
    if (options_.segType == HYBM_MST_DRAM) {
        aclrtMemLocation loc;
        loc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        loc.id = options_.rankId;
        aclrtMemAccessDesc des;
        des.location = loc;
        des.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
        auto localVirtualBase = reservedVirtualAddresses_[options_.rankId];
        ACLSHMEM_CHECK_RET(aclrtMemSetAccess(reinterpret_cast<void *>(localVirtualBase), options_.size, &des, 1),
            "aclrtMemSetAccess failed.", ACLSHMEM_SMEM_ERROR);
    }
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::UnReserveMemorySpace() noexcept
{
    SHM_LOG_INFO("un-reserve memory space.");
    FreeMemory();
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::AllocLocalMemory(uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept
{
    if ((size % DEVICE_LARGE_PAGE_SIZE) != 0UL || size + allocatedSize_ > options_.size) {
        SHM_LOG_ERROR("invalid allocate memory size : " << size << ", now used " << allocatedSize_ << " of "
                                                       << options_.size);
        return ACLSHMEM_INVALID_PARAM;
    }
    auto localVirtualBase = reservedVirtualAddresses_[options_.rankId];
    aclrtPhysicalMemProp prop;
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = (options_.segType == HYBM_MST_DRAM) ? ACL_DDR_MEM_P2P_HUGE : ACL_HBM_MEM_HUGE;
    prop.location.id = (options_.segType == HYBM_MST_DRAM) ? -1 : deviceId_;
    prop.location.type = (options_.segType == HYBM_MST_DRAM) ? ACL_MEM_LOCATION_TYPE_HOST_NUMA : ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.reserve = 0;
    auto ret = aclrtMallocPhysical(&local_handle_, size, &prop, 0);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(options_.rankId << " aclrtMallocPhysical memory failed: " << ret << " segType: " << (int)options_.segType);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    SHM_LOG_DEBUG(options_.rankId << " aclrtMallocPhysical memory success size: " << size << " vaddr: " << reinterpret_cast<void *>(localVirtualBase) << " segType: " << (int)options_.segType);
    ACLSHMEM_CHECK_RET(aclrtMapMem(reinterpret_cast<void *>(localVirtualBase), size, 0, local_handle_, 0));

    if (SetMemAccess() != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_SMEM_ERROR;
    }

    slice = std::make_shared<MemSlice>(sliceCount_++, MEM_TYPE_DEVICE_HBM, MEM_PT_TYPE_SVM,
                                       reinterpret_cast<uint64_t>(localVirtualBase), size);
    slices_.emplace(slice->index_, slice);
    SHM_LOG_DEBUG("allocate slice(idx:" << slice->index_ << ", size:" << slice->size_ << ").");
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::RegisterMemory(const void *addr, uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept
{
    SHM_LOG_INFO("MemSegmentDevice NOT SUPPORT RegisterMemory");
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::ReleaseSliceMemory(const std::shared_ptr<MemSlice> &slice) noexcept
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
    auto res = aclrtUnmapMem(reinterpret_cast<void *>(slice->vAddress_));
    SHM_LOG_INFO("unmap slice(idx:" << slice->index_ << "), size: " << slice->size_ << " return:" << res);

    slices_.erase(pos);
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::Export(std::string &exInfo) noexcept
{
    SHM_LOG_INFO("MemSegmentDevice not supported export device info.");
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::Export(const std::shared_ptr<MemSlice> &slice, std::string &exInfo) noexcept
{
    if (slice == nullptr) {
        SHM_LOG_ERROR("input slice is nullptr");
        return ACLSHMEM_INVALID_PARAM;
    }
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip export");
        return ACLSHMEM_SUCCESS;
    }
    auto pos = slices_.find(slice->index_);
    if (pos == slices_.end()) {
        SHM_LOG_ERROR("input slice(idx:" << slice->index_ << ") not exist.");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (pos->second.slice != slice) {
        SHM_LOG_ERROR("input slice(magic:" << std::hex << slice->magic_ << ") not match.");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto exp = exportMap_.find(slice->index_);
    if (exp != exportMap_.end()) {
        exInfo = exp->second;
        return ACLSHMEM_SUCCESS;
    }

    HbmExportInfo info{};
    auto handle_type = ACL_MEM_SHARE_HANDLE_TYPE_FABRIC;
    auto ret = aclrtMemExportToShareableHandleV2(local_handle_, 1, handle_type, &info.shareHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("aclrtMemExportToShareableHandleV2 failed, ret: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    auto localVirtualBase = reservedVirtualAddresses_[options_.rankId];
    info.mappingOffset = slice->vAddress_ - (uint64_t)(ptrdiff_t)localVirtualBase;
    info.sliceIndex = static_cast<uint32_t>(slice->index_);
    info.deviceId = options_.devId;
    info.logicDeviceId = logicDeviceId_;
    info.pid = pid_;
    info.rankId = options_.rankId;
    info.size = slice->size_;
    info.entityId = entityId_;
    info.sdid = sdid_;
    info.serverId = serverId_;
    info.superPodId = superPodId_;
    info.pageTblType = MEM_PT_TYPE_SVM;
    info.memSegType = options_.segType;
    info.exchangeType = HYBM_INFO_EXG_IN_NODE;
    info.magic = (options_.segType == HYBM_MST_DRAM) ? DRAM_SLICE_EXPORT_INFO_MAGIC : HBM_SLICE_EXPORT_INFO_MAGIC;
    ret = LiteralExInfoTranslater<HbmExportInfo>{}.Serialize(info, exInfo);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("export info failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    exportMap_[slice->index_] = exInfo;
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::GetExportSliceSize(size_t &size) noexcept
{
    size = sizeof(HbmExportInfo);
    return ACLSHMEM_SUCCESS;
}

// import可重入
Result MemSegmentDevice::Import(const std::vector<std::string> &allExInfo, void *addresses[]) noexcept
{
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip import");
        return ACLSHMEM_SUCCESS;
    }
    std::map<uint16_t, HbmExportInfo> importMap;
    LiteralExInfoTranslater<HbmExportInfo> translator;
    std::vector<HbmExportInfo> desInfos{allExInfo.size()};
    for (auto i = 0U; i < allExInfo.size(); i++) {
        auto ret = translator.Deserialize(allExInfo[i], desInfos[i]);
        if (ret != 0) {
            SHM_LOG_ERROR("deserialize imported info(" << i << ") failed.");
            return ACLSHMEM_INVALID_PARAM;
        }
        importMap.emplace(desInfos[i].rankId, desInfos[i]);
    }
    importMap_ = std::move(importMap);
    std::copy(desInfos.begin(), desInfos.end(), std::back_inserter(imports_));
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::Mmap() noexcept
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
        if (mappedMem_.find((uint64_t)remoteAddress) != mappedMem_.end()) {
            SHM_LOG_INFO("remote slice on rank(" << im.rankId << ") has maped: " << (void *)remoteAddress);
            continue;
        }

        if (!CanMapRemote(im)) {
            SHM_LOG_INFO("remote slice on rank(" << im.rankId << ") SDMA cannot reaches.");
            continue;
        }

        SHM_LOG_DEBUG("remote slice on rank(" << im.rankId << ") should map to: " << (void *)remoteAddress
                                             << ", size = " << im.size);
        aclrtDrvMemHandle imported_handle;
        auto handle_type = ACL_MEM_SHARE_HANDLE_TYPE_FABRIC;
        ACLSHMEM_CHECK_RET(aclrtMemImportFromShareableHandleV2(&im.shareHandle, handle_type, 0, &imported_handle),
            "aclrtMemImportFromShareableHandleV2 failed", ACLSHMEM_SMEM_ERROR);
        auto ret = aclrtMapMem(reinterpret_cast<void *>(remoteAddress), im.size, 0, imported_handle, 0);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("aclrtMapMem memory failed:" << ret);
            return ACLSHMEM_DL_FUNC_FAILED;
        }
        mappedMem_.insert((uint64_t)remoteAddress);
    }
    imports_.clear();
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::Unmap() noexcept
{
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip unmap");
        return ACLSHMEM_SUCCESS;
    }
    for (auto va : mappedMem_) {
        int32_t ret = aclrtUnmapMem(reinterpret_cast<void *>(va));
        if (ret != 0) {
            SHM_LOG_ERROR("aclrtUnmapMem memory failed:" << ret);
        }
    }
    mappedMem_.clear();
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::RemoveImported(const std::vector<uint32_t> &ranks) noexcept
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

    for (auto &rank : ranks) {
        size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
        uint64_t addr = reinterpret_cast<uint64_t>(globalVirtualAddress_) + reserveAlignedSize * rank;
        auto it = mappedMem_.lower_bound(addr);
        auto st = it;
        while (it != mappedMem_.end() && (*it) < addr + reserveAlignedSize) {
            int32_t ret = aclrtUnmapMem(reinterpret_cast<void *>(*it));
            if (ret != 0) {
                SHM_LOG_ERROR("aclrtUnmapMem failed " << ret);
            }
            it++;
        }

        if (st != it) {
            mappedMem_.erase(st, it);
        }
    }
    return 0;
}

std::shared_ptr<MemSlice> MemSegmentDevice::GetMemSlice(hybm_mem_slice_t slice) const noexcept
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

bool MemSegmentDevice::MemoryInRange(const void *begin, uint64_t size) const noexcept
{
    if (begin < globalVirtualAddress_) {
        return false;
    }

    if (static_cast<const uint8_t *>(begin) + size >= globalVirtualAddress_ + totalVirtualSize_) {
        return false;
    }

    return true;
}

void MemSegmentDevice::FreeMemory() noexcept
{
    while (!slices_.empty()) {
        auto slice = slices_.begin()->second.slice;
        ReleaseSliceMemory(slice);
    }

    if (local_handle_ != nullptr) {
        auto ret = aclrtFreePhysical(local_handle_);
        if (ret != 0) {
            SHM_LOG_ERROR("aclrtFreePhysical failed. ret: " << ret);
            return;
        }
    }
    
    allocatedSize_ = 0;
    sliceCount_ = 0;
    if (globalVirtualAddress_ != nullptr) {
        if (socType_ == AscendSocType::ASCEND_950 && !reservedVirtualAddresses_.empty()) {
            aclrtReleaseMemAddress(reinterpret_cast<void *>(reservedVirtualAddresses_[0]));
        } else {
            for (auto reserved : reservedVirtualAddresses_) {
                aclrtReleaseMemAddress(reinterpret_cast<void *>(reserved));
            }
        }
        reservedVirtualAddresses_.clear();
        totalVirtualSize_ = 0;
        globalVirtualAddress_ = nullptr;
    }
}

bool MemSegmentDevice::CanMapRemote(const HbmExportInfo &rmi) noexcept
{
    return IsSdmaAccessible(rmi.superPodId, rmi.serverId, rmi.logicDeviceId);
}

void MemSegmentDevice::GetDeviceInfo(uint32_t &sdId, uint32_t &serverId, uint32_t &superPodId) noexcept
{
    sdId = sdid_;
    serverId = serverId_;
    superPodId = superPodId_;
}

bool MemSegmentDevice::GetRankIdByAddr(const void *addr, uint64_t size, uint32_t &rankId) const noexcept
{
    return true;
}

bool MemSegmentDevice::CheckSdmaReaches(uint32_t remoteRankId) const noexcept
{
    auto pos = importMap_.find(static_cast<uint16_t>(remoteRankId));
    if (pos == importMap_.end()) {
        return false;
    }

    return IsSdmaAccessible(pos->second.superPodId, pos->second.serverId, pos->second.logicDeviceId);
}
}  // namespace shm

#endif