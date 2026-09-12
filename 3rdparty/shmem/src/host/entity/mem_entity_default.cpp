/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "shmemi_logger.h"
#include "dl_api.h"
#include "dl_acl_api.h"
#include "hybm_device_mem_segment.h"
#include "hybm_ex_info_transfer.h"
#include "mem_entity_default.h"
#include "mem_entity_inter.h"

using namespace shm::transport;

namespace shm {

thread_local bool MemEntityDefault::isSetDevice_ = false;

MemEntityDefault::MemEntityDefault(int id) noexcept : id_(id), initialized(false) {}

MemEntityDefault::~MemEntityDefault()
{
    SHM_LOG_INFO("Deconstruct MemEntity begin, try to release resource.");
    ReleaseResources();
}

int32_t MemEntityDefault::Initialize(const hybm_options *options) noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized) {
        SHM_LOG_WARN("The MemEntity has already been initialized, no action needs.");
        return ACLSHMEM_SUCCESS;
    }
    SHM_VALIDATE_RETURN((id_ >= 0 && (uint32_t)(id_) < HYBM_ENTITY_NUM_MAX),
                       "input entity id is invalid, input: " << id_ << " must be less than: " << HYBM_ENTITY_NUM_MAX,
                       ACLSHMEM_INVALID_PARAM);

    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(CheckOptions(options), "check options failed.");

    options_ = *options;

    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(LoadExtendLibrary(), "LoadExtendLibrary failed.");
    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(InitSegment(), "InitSegment failed.");

    auto ret = InitTransManager();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("init transport manager failed");
        return ret;
    }

    initialized = true;
    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::SetThreadAclDevice()
{
    if (isSetDevice_) {
        return ACLSHMEM_SUCCESS;
    }
    auto ret = DlAclApi::AclrtSetDevice(HybmGetInitDeviceId());
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Set device id to be " << HybmGetInitDeviceId() << " failed: " << ret);
        return ret;
    }
    isSetDevice_ = true;
    SHM_LOG_DEBUG("Set device id to be " << HybmGetInitDeviceId() << " success.");
    return ACLSHMEM_SUCCESS;
}

void MemEntityDefault::UnInitialize() noexcept
{
    SHM_LOG_INFO("MemEntity UnInitialize begin, try to release resource.");
    ReleaseResources();
}

int32_t MemEntityDefault::ReserveMemorySpace(void **reservedMem) noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized) {
        SHM_LOG_INFO("the object is not initialized, please check whether Initialize is called.");
        return ACLSHMEM_NOT_INITED;
    }
    if (hbmSegment_ != nullptr) {
        auto ret = hbmSegment_->ReserveMemorySpace(&hbmGva_);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("Failed to reserver HBM memory space ret: " << ret);
            return ret;
        }
        *reservedMem = hbmGva_;
    }
    if (dramSegment_ != nullptr) {
        auto ret = dramSegment_->ReserveMemorySpace(&dramGva_);
        if (ret != ACLSHMEM_SUCCESS) {
            UnReserveMemorySpace();
            SHM_LOG_ERROR("Failed to reserver DRAM memory space ret: " << ret);
            return ret;
        }
        *reservedMem = dramGva_;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::UnReserveMemorySpace() noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized) {
        SHM_LOG_INFO("the object is not initialized, please check whether Initialize is called.");
        return ACLSHMEM_NOT_INITED;
    }

    if (transportManager_) {
        transportManager_->CloseDevice();
        transportManager_ = nullptr;
    }

    if (hbmSegment_ != nullptr) {
        hbmSegment_->UnReserveMemorySpace();
        hbmGva_ = nullptr;
    }
    if (dramSegment_ != nullptr) {
        dramSegment_->UnReserveMemorySpace();
        dramGva_ = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::AllocLocalMemory(uint64_t size, hybm_mem_type mType, uint32_t flags, hybm_mem_slice_t &slice) noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized) {
        SHM_LOG_INFO("the object is not initialized, please check whether Initialize is called.");
        return ACLSHMEM_NOT_INITED;
    }

    if ((size % DEVICE_LARGE_PAGE_SIZE) != 0) {
        SHM_LOG_ERROR("allocate memory size: " << size << " invalid, page size is: " << DEVICE_LARGE_PAGE_SIZE);
        return ACLSHMEM_INVALID_PARAM;
    }

    auto segment = mType == HYBM_MEM_TYPE_DEVICE ? hbmSegment_ : dramSegment_;
    if (segment == nullptr) {
        SHM_LOG_ERROR("allocate memory with mType: " << mType << ", no segment.");
        return ACLSHMEM_INVALID_PARAM;
    }

    std::shared_ptr<MemSlice> realSlice;
    auto ret = segment->AllocLocalMemory(size, realSlice);
    if (ret != 0) {
        SHM_LOG_ERROR("segment allocate slice with size: " << size << " failed: " << ret);
        return ret;
    }

    slice = realSlice->ConvertToId();
    transport::TransportMemoryRegion info;
    info.size = realSlice->size_;
    info.addr = realSlice->vAddress_;
    info.access = transport::REG_MR_ACCESS_FLAG_BOTH_READ_WRITE;
    info.flags = 
        segment->GetMemoryType() == HYBM_MEM_TYPE_DEVICE ? transport::REG_MR_FLAG_HBM : transport::REG_MR_FLAG_DRAM;
    if (transportManager_ != nullptr) {
        ret = transportManager_->RegisterMemoryRegion(info);
        if (ret != 0) {
            SHM_LOG_ERROR("register memory region allocate failed: " << ret << ", info: " << info);
            auto res = segment->ReleaseSliceMemory(realSlice);
            if (res != ACLSHMEM_SUCCESS) {
                SHM_LOG_ERROR("failed to release slice memory: " << res);
            }
            return ret;
        }
    }

    return UpdateHybmDeviceInfo(0);
}

int32_t MemEntityDefault::RegisterLocalMemory(const void *ptr, uint64_t size, uint32_t flags,
                                              hybm_mem_slice_t &slice) noexcept
{
    if (ptr == nullptr || size == 0) {
        SHM_LOG_ERROR("input ptr or size(" << size << ") is invalid");
        return ACLSHMEM_INVALID_PARAM;
    }

    if ((size % DEVICE_LARGE_PAGE_SIZE) != 0) {
        uint64_t originalSize = size;
        size = ((size + DEVICE_LARGE_PAGE_SIZE - 1) / DEVICE_LARGE_PAGE_SIZE) * DEVICE_LARGE_PAGE_SIZE;
        SHM_LOG_INFO("size: " << originalSize << " not aligned to large page (" << DEVICE_LARGE_PAGE_SIZE <<
                    "), rounded to: " << size);
    }

    auto addr = static_cast<uint64_t>(reinterpret_cast<ptrdiff_t>(ptr));
    bool isHbm = (addr >= HYBM_DEVICE_VA_START && addr < HYBM_DEVICE_VA_END);
    SHM_LOG_INFO("Hbm: " << isHbm << std::hex << ", addrs: 0x" << addr
                        << ", start: 0x" << HYBM_DEVICE_VA_START << ", end: 0x" << HYBM_DEVICE_VA_END);
    std::shared_ptr<MemSegment> segment = nullptr;
    if (dramSegment_ == nullptr) {
        segment = hbmSegment_;
    } else {
        segment = dramSegment_;
    }
    SHM_VALIDATE_RETURN(segment != nullptr, "address for segment is null.", ACLSHMEM_INVALID_PARAM);

    std::shared_ptr<MemSlice> realSlice;
    auto ret = segment->RegisterMemory(ptr, size, realSlice);
    if (ret != 0) {
        SHM_LOG_ERROR("segment register slice with size: " << size << " failed: " << ret);
        return ret;
    }

    if (transportManager_ != nullptr) {
        transport::TransportMemoryRegion mr;
        mr.addr = (uint64_t)(ptrdiff_t)ptr;
        mr.size = size;
        mr.flags = (isHbm ? transport::REG_MR_FLAG_HBM : transport::REG_MR_FLAG_DRAM);
        ret = transportManager_->RegisterMemoryRegion(mr);
        if (ret != 0) {
            SHM_LOG_ERROR("register MR: " << mr << " to transport failed: " << ret);
            return ret;
        }
    }

    slice = realSlice->ConvertToId();
    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::FreeLocalMemory(hybm_mem_slice_t slice, uint32_t flags) noexcept
{
    if (!initialized) {
        SHM_LOG_INFO("the object is not initialized, please check whether Initialize is called.");
        return ACLSHMEM_INVALID_PARAM;
    }

    std::shared_ptr<MemSlice> memSlice;
    if (hbmSegment_ != nullptr && (memSlice = hbmSegment_->GetMemSlice(slice)) != nullptr) {
        hbmSegment_->ReleaseSliceMemory(memSlice);
    } else if (dramSegment_ != nullptr && (memSlice = dramSegment_->GetMemSlice(slice)) != nullptr) {
        dramSegment_->ReleaseSliceMemory(memSlice);
    }

    if (transportManager_ != nullptr && memSlice != nullptr) {
        auto ret = transportManager_->UnregisterMemoryRegion(memSlice->vAddress_);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("UnregisterMemoryRegion failed, please check input slice.");
        }
    }
    SHM_LOG_DEBUG("free local memory successed.");
    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::ExportExchangeInfo(ExchangeInfoWriter &desc, uint32_t flags) noexcept
{
    if (!initialized) {
        SHM_LOG_INFO("the object is not initialized, please check whether Initialize is called.");
        return ACLSHMEM_NOT_INITED;
    }

    std::string info;
    EntityExportInfo exportInfo;
    exportInfo.version = EXPORT_INFO_VERSION;
    exportInfo.rankId = options_.rankId;
    exportInfo.role = static_cast<uint16_t>(options_.role);
    if (transportManager_ != nullptr) {
        auto &nic = transportManager_->GetNic();
        if (nic.size() >= sizeof(exportInfo.nic)) {
            SHM_LOG_ERROR("transport get nic(" << nic << ") too long.");
            return ACLSHMEM_INNER_ERROR;
        }
        size_t copyLen = std::min(nic.size(), sizeof(exportInfo.nic));
        std::copy_n(nic.c_str(), copyLen, exportInfo.nic);
        auto ret = LiteralExInfoTranslater<EntityExportInfo>{}.Serialize(exportInfo, info);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("export info failed: " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
    }

    auto ret = desc.Append(info.data(), info.size());
    if (ret != 0) {
        SHM_LOG_ERROR("export to string wrong size: " << info.size());
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::ExportExchangeInfo(hybm_mem_slice_t slice, ExchangeInfoWriter &desc, uint32_t flags) noexcept
{
    if (!initialized) {
        SHM_LOG_INFO("the object is not initialized, please check whether Initialize is called.");
        return ACLSHMEM_NOT_INITED;
    }
    if (slice == nullptr) {
        return ExportWithoutSlice(desc, flags);
    }

    return ExportWithSlice(slice, desc, flags);
}

int32_t MemEntityDefault::ImportExchangeInfo(const ExchangeInfoReader desc[], uint32_t count, void *addresses[],
                                             uint32_t flags) noexcept
{
    if (!initialized) {
        SHM_LOG_ERROR("the object is not initialized, please check whether Initialize is called.");
        return ACLSHMEM_NOT_INITED;
    }

    auto ret = SetThreadAclDevice();
    if (ret != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_INNER_ERROR;
    }

    if (desc == nullptr) {
        SHM_LOG_ERROR("the input desc is nullptr.");
        return ACLSHMEM_INNER_ERROR;
    }

    std::unordered_map<uint32_t, std::vector<transport::TransportMemoryKey>> tempKeyMap;
    ret = ImportForTransport(desc, count, flags);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    if (desc[0].LeftBytes() == 0) {
        SHM_LOG_INFO("no segment need import.");
        return ACLSHMEM_SUCCESS;
    }

    uint64_t magic;
    if (desc[0].Test(magic) < 0) {
        SHM_LOG_ERROR("left import data no magic size.");
        return ACLSHMEM_SUCCESS;
    }

    auto currentSegment = magic == DRAM_SLICE_EXPORT_INFO_MAGIC ? dramSegment_ : hbmSegment_;
    std::vector<std::string> infos;
    for (auto i = 0U; i < count; i++) {
        infos.emplace_back(desc[i].LeftToString());
    }

    ret = currentSegment->Import(infos, addresses);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("segment import infos failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::GetExportSliceInfoSize(size_t &size) noexcept
{
    size_t exportSize = 0;
    auto segment = hbmSegment_ == nullptr ? dramSegment_ : hbmSegment_;
    if (segment == nullptr) {
        SHM_LOG_ERROR("segment is null.");
        return ACLSHMEM_INNER_ERROR;
    }
    auto ret = segment->GetExportSliceSize(exportSize);
    if (ret != 0) {
        SHM_LOG_ERROR("GetExportSliceSize for segment failed: " << ret);
        return ret;
    }

    if (transportManager_ != nullptr) {
        exportSize += sizeof(SliceExportTransportKey);
    }
    size = exportSize;
    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::SetExtraContext(const void *context, uint32_t size) noexcept
{
    if (!initialized) {
        SHM_LOG_INFO("the object is not initialized, please check whether Initialize is called.");
        return ACLSHMEM_NOT_INITED;
    }

    SHM_ASSERT_RETURN(context != nullptr, ACLSHMEM_INVALID_PARAM);
    if (size > HYBM_DEVICE_USER_CONTEXT_PRE_SIZE) {
        SHM_LOG_ERROR("set extra context failed, context size is too large: " << size << " limit: "
                                                                             << HYBM_DEVICE_USER_CONTEXT_PRE_SIZE);
        return ACLSHMEM_INVALID_PARAM;
    }

    uint64_t addr = HYBM_DEVICE_USER_CONTEXT_ADDR + id_ * HYBM_DEVICE_USER_CONTEXT_PRE_SIZE;
    SHM_LOG_DEBUG("set extra context to addr: 0x" << std::hex << addr << " size: " << size);
    auto ret = DlAclApi::AclrtMemcpy((void *)addr, HYBM_DEVICE_USER_CONTEXT_PRE_SIZE, context, size,
                                     ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("memcpy user context failed, ret: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    return UpdateHybmDeviceInfo(size);
}

void MemEntityDefault::Unmap() noexcept
{
    if (!initialized) {
        SHM_LOG_INFO("the object is not initialized, please check whether Initialize is called.");
        return;
    }

    if (hbmSegment_ != nullptr) {
        hbmSegment_->Unmap();
    }
    if (dramSegment_ != nullptr) {
        dramSegment_->Unmap();
    }
}

int32_t MemEntityDefault::Mmap() noexcept
{
    if (!initialized) {
        SHM_LOG_INFO("the object is not initialized, please check whether Initialize is called.");
        return ACLSHMEM_NOT_INITED;
    }

    if (hbmSegment_ != nullptr) {
        auto ret = hbmSegment_->Mmap();
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }

    if (dramSegment_ != nullptr) {
        auto ret = dramSegment_->Mmap();
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::RemoveImported(const std::vector<uint32_t> &ranks) noexcept
{
    if (!initialized) {
        SHM_LOG_INFO("the object is not initialized, please check whether Initialize is called.");
        return ACLSHMEM_NOT_INITED;
    }

    if (hbmSegment_ != nullptr) {
        auto ret = hbmSegment_->RemoveImported(ranks);
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }

    if (dramSegment_ != nullptr) {
        auto ret = dramSegment_->RemoveImported(ranks);
        if (ret != ACLSHMEM_SUCCESS) {
            return ret;
        }
    }

    return ACLSHMEM_SUCCESS;
}

bool MemEntityDefault::CheckAddressInEntity(const void *ptr, uint64_t length) const noexcept
{
    if (!initialized) {
        SHM_LOG_ERROR("the object is not initialized, please check whether Initialize is called.");
        return false;
    }

    if (hbmSegment_ != nullptr && hbmSegment_->MemoryInRange(ptr, length)) {
        return true;
    }

    if (dramSegment_ != nullptr && dramSegment_->MemoryInRange(ptr, length)) {
        return true;
    }

    return false;
}

int MemEntityDefault::CheckOptions(const hybm_options *options) noexcept
{
    if (options == nullptr) {
        SHM_LOG_ERROR("initialize with nullptr.");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (options->rankId >= options->rankCount) {
        SHM_LOG_ERROR("local rank id: " << options->rankId << " invalid, total is " << options->rankCount);
        return ACLSHMEM_INVALID_PARAM;
    }

    return ACLSHMEM_SUCCESS;
}

int MemEntityDefault::LoadExtendLibrary() noexcept
{
    if (options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_RDMA) {
        auto ret = DlApi::LoadExtendLibrary(DL_EXT_LIB_DEVICE_RDMA);
        if (ret != 0) {
            SHM_LOG_ERROR("LoadExtendLibrary for DEVICE RDMA failed: " << ret);
            return ret;
        }
    }

    if (options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_SDMA) {
        auto ret = DlApi::LoadExtendLibrary(DL_EXT_LIB_DEVICE_SDMA);
        if (ret != 0) {
            SHM_LOG_ERROR("LoadExtendLibrary for DEVICE SDMA failed: " << ret);
            return ret;
        }
    }

    if (options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_UDMA) {
#if defined(ACLSHMEM_UDMA_SUPPORT)
        auto ret = DlApi::LoadExtendLibrary(DL_EXT_LIB_DEVICE_UDMA);
        if (ret != 0) {
            SHM_LOG_ERROR("LoadExtendLibrary for DEVICE UDMA failed: " << ret);
            return ret;
        }
#else
        SHM_LOG_ERROR("DEVICE UDMA support is not enabled in this build.");
        return ACLSHMEM_NOT_SUPPORTED;
#endif
    }

    return ACLSHMEM_SUCCESS;
}

int MemEntityDefault::UpdateHybmDeviceInfo(uint32_t extCtxSize) noexcept
{
    HybmDeviceMeta info;
    auto addr = HYBM_DEVICE_META_ADDR + HYBM_DEVICE_GLOBAL_META_SIZE + id_ * HYBM_DEVICE_PRE_META_SIZE;

    SetHybmDeviceInfo(info);
    info.extraContextSize = extCtxSize;
    auto ret = DlAclApi::AclrtMemcpy((void *)addr, DEVICE_LARGE_PAGE_SIZE, &info, sizeof(HybmDeviceMeta),
                                     ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("update hybm info memory failed, ret: " << ret << ", addr:" << (void *)addr);
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_DEBUG("update hybm info memory success, addr:" << (void *)addr);
    return ACLSHMEM_SUCCESS;
}

void MemEntityDefault::SetHybmDeviceInfo(HybmDeviceMeta &info)
{
    info.entityId = id_;
    info.rankId = options_.rankId;
    info.rankSize = options_.rankCount;
    info.symmetricSize = options_.deviceVASpace;
    info.extraContextSize = 0;
    if (transportManager_ != nullptr) {
        info.qpInfoAddress = (uint64_t)(ptrdiff_t)transportManager_->GetQpInfo();
    } else {
        info.qpInfoAddress = 0UL;
    }
}

int32_t MemEntityDefault::ExportWithSlice(hybm_mem_slice_t slice, ExchangeInfoWriter &desc, uint32_t flags) noexcept
{
    uint64_t exportMagic = 0;
    std::string info;
    std::shared_ptr<MemSlice> realSlice;
    std::shared_ptr<MemSegment> currentSegment;
    if (hbmSegment_ != nullptr) {
        realSlice = hbmSegment_->GetMemSlice(slice);
        currentSegment = hbmSegment_;
        exportMagic = HBM_SLICE_EXPORT_INFO_MAGIC;
    }
    if (realSlice == nullptr && dramSegment_ != nullptr) {
        realSlice = dramSegment_->GetMemSlice(slice);
        currentSegment = dramSegment_;
        exportMagic = DRAM_SLICE_EXPORT_INFO_MAGIC;
    }
    if (realSlice == nullptr) {
        SHM_LOG_ERROR("cannot find input slice for export.");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto ret = currentSegment->Export(realSlice, info);
    if (ret != 0) {
        SHM_LOG_ERROR("export to string failed: " << ret);
        return ret;
    }

    if (transportManager_ != nullptr) {
        SliceExportTransportKey transportKey{exportMagic, options_.rankId, realSlice->vAddress_};
        ret = transportManager_->QueryMemoryKey(realSlice->vAddress_, transportKey.key);
        if (ret != 0) {
            SHM_LOG_ERROR("query memory key when export slice failed: " << ret);
            return ret;
        }
        ret = desc.Append(transportKey);
        if (ret != 0) {
            SHM_LOG_ERROR("append transport key failed: " << ret);
            return ret;
        }
    }
    static std::mutex debug_mutex;
    std::lock_guard<std::mutex> lock(debug_mutex);
    int status = desc.Append(info.data(), info.size());
    if (status != 0) {
        SHM_LOG_ERROR("export to string wrong size: " << info.size() << " ret: " << status);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::ExportWithoutSlice(ExchangeInfoWriter &desc, uint32_t flags) noexcept
{
    std::string info;
    int32_t ret = ACLSHMEM_INNER_ERROR;
    ret = ExportExchangeInfo(desc, flags);
    if (ret != 0) {
        SHM_LOG_ERROR("ExportExchangeInfo failed: " << ret);
        return ret;
    }

    ret = desc.Append(info.data(), info.size());
    if (ret != 0) {
        SHM_LOG_ERROR("export to string wrong size: " << info.size());
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::ImportForTransportPrecheck(const ExchangeInfoReader desc[],
                                                     uint32_t &count,
                                                     bool &importInfoEntity)
{
    int ret = ACLSHMEM_SUCCESS;
    uint64_t magic;
    EntityExportInfo entityExportInfo;
    SliceExportTransportKey transportKey;
    for (auto i = 0U; i < count; i++) {
        ret = desc[i].Test(magic);
        if (ret != 0) {
            SHM_LOG_ERROR("read magic from import : " << i << " failed.");
            return ACLSHMEM_INNER_ERROR;
        }

        if (magic == ENTITY_EXPORT_INFO_MAGIC) {
            ret = desc[i].Read(entityExportInfo);
            if (ret == 0) {
                importedRanks_[entityExportInfo.rankId] = entityExportInfo;
                importInfoEntity = true;
            }
        } else if (magic == DRAM_SLICE_EXPORT_INFO_MAGIC || magic == HBM_SLICE_EXPORT_INFO_MAGIC) {
            ret = desc[i].Read(transportKey);
            if (ret == 0) {
                std::unique_lock<std::mutex> uniqueLock{importMutex_};
                importedMemories_[transportKey.rankId][transportKey.address] = transportKey.key;
            }
        } else {
            SHM_LOG_ERROR("magic(" << std::hex << magic << ") invalid");
            ret = ACLSHMEM_INNER_ERROR;
        }

        if (ret != 0) {
            SHM_LOG_ERROR("read info for transport failed: " << ret);
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

int32_t MemEntityDefault::ImportForTransport(const ExchangeInfoReader desc[], uint32_t count, uint32_t flags) noexcept
{
    if (transportManager_ == nullptr) {
        return ACLSHMEM_SUCCESS;
    }

    int ret = ACLSHMEM_SUCCESS;
    bool importInfoEntity = false;
    ret = ImportForTransportPrecheck(desc, count, importInfoEntity);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    transport::HybmTransPrepareOptions transOptions;
    std::unique_lock<std::mutex> uniqueLock{importMutex_};
    for (auto &rank : importedRanks_) {
        if (options_.role != HYBM_ROLE_PEER && static_cast<hybm_role_type>(rank.second.role) == options_.role) {
            continue;
        }

        transOptions.options[rank.first].role = static_cast<hybm_role_type>(rank.second.role);
        transOptions.options[rank.first].nic = rank.second.nic;
    }
    for (auto &mr : importedMemories_) {
        auto pos = transOptions.options.find(mr.first);
        if (pos != transOptions.options.end()) {
            for (auto &key : mr.second) {
                pos->second.memKeys.emplace_back(key.second);
            }
        }
    }
    uniqueLock.unlock();

    if (options_.role != HYBM_ROLE_PEER || importInfoEntity) {
        ret = transportManager_->ConnectWithOptions(transOptions);
        if (ret != 0) {
            SHM_LOG_ERROR("Transport Manager ConnectWithOptions failed: " << ret);
            return ret;
        }
        if (importInfoEntity) {
            return UpdateHybmDeviceInfo(0);
        }
    }

    return ACLSHMEM_SUCCESS;
}

Result MemEntityDefault::InitSegment()
{
    SHM_LOG_DEBUG("Initialize segment with type: " << std::hex << options_.memType);
    if (options_.memType & HYBM_MEM_TYPE_DEVICE) {
        auto ret = InitHbmSegment();
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("InitHbmSegment() failed: " << ret);
            return ret;
        }
    }

    if (options_.memType & HYBM_MEM_TYPE_HOST) {
        auto ret = InitDramSegment();
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("InitDramSegment() failed: " << ret);
            return ret;
        }
    }
    return ACLSHMEM_SUCCESS;
}

Result MemEntityDefault::InitHbmSegment()
{
    MemSegmentOptions segmentOptions;
    if (options_.deviceVASpace == 0) {
        SHM_LOG_INFO("Hbm rank space is zero.");
        return ACLSHMEM_SUCCESS;
    }
    segmentOptions.size = options_.deviceVASpace;
    segmentOptions.segType = HYBM_MST_HBM;
    segmentOptions.devId = HybmGetInitDeviceId();
    segmentOptions.role = options_.role;
    segmentOptions.dataOpType = options_.bmDataOpType;
    segmentOptions.rankId = options_.rankId;
    segmentOptions.rankCnt = options_.rankCount;
    hbmSegment_ = MemSegment::Create(segmentOptions, id_);
    SHM_VALIDATE_RETURN(hbmSegment_ != nullptr, "create segment failed", ACLSHMEM_INVALID_PARAM);
    return ACLSHMEM_SUCCESS;
}

Result MemEntityDefault::InitDramSegment()
{
    if (options_.hostVASpace == 0) {
        SHM_LOG_INFO("Dram rank space is zero.");
        return ACLSHMEM_SUCCESS;
    }

    MemSegmentOptions segmentOptions;
    segmentOptions.size = options_.hostVASpace;
    segmentOptions.devId = HybmGetInitDeviceId();
    segmentOptions.segType = HYBM_MST_DRAM;
    segmentOptions.rankId = options_.rankId;
    segmentOptions.rankCnt = options_.rankCount;
    segmentOptions.dataOpType = options_.bmDataOpType;
    segmentOptions.flags = options_.flags;
    if (options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_RDMA) {
        segmentOptions.shared = false;
    }
    dramSegment_ = MemSegment::Create(segmentOptions, id_);
    if (dramSegment_ == nullptr) {
        SHM_LOG_ERROR("Failed to create dram segment");
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

Result MemEntityDefault::InitTransManager()
{
    if (options_.rankCount <= 1) {
        SHM_LOG_INFO("rank total count : " << options_.rankCount << ", no transport.");
        return ACLSHMEM_SUCCESS;
    }
    if (((options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_RDMA) == 0) &&
        ((options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_SDMA) == 0) &&
        ((options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_UDMA) == 0)) {
        SHM_LOG_DEBUG("Data operator type mathchs none of RDMA, SDMA or UDMA, skip transport init.");
        return ACLSHMEM_SUCCESS;
    }

    if (options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_RDMA) {
        transportManager_ = transport::TransportManager::Create(TransportType::TT_HCCP);
    } else if (options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_SDMA) {
        transportManager_ = transport::TransportManager::Create(TransportType::TT_SDMA);
    } else if (options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_UDMA) {
#if defined(ACLSHMEM_UDMA_SUPPORT)
        transportManager_ = transport::TransportManager::Create(TransportType::TT_UDMA);
#else
        SHM_LOG_ERROR("DEVICE UDMA support is not enabled in this build.");
        return ACLSHMEM_NOT_SUPPORTED;
#endif
    }

    transport::TransportOptions options;
    options.rankId = options_.rankId;
    options.rankCount = options_.rankCount;
    options.protocol = options_.bmDataOpType;
    options.role = options_.role;
    options.nic = options_.nic;
    auto ret = transportManager_->OpenDevice(options);
    if (ret != 0) {
        SHM_LOG_ERROR("Failed to open device, ret: " << ret);
        transportManager_ = nullptr;
    }
    return ret;
}

bool MemEntityDefault::SdmaReaches(uint32_t remoteRank) const noexcept
{
    if (hbmSegment_ != nullptr) {
        return hbmSegment_->CheckSdmaReaches(remoteRank);
    }

    if (dramSegment_ != nullptr) {
        return dramSegment_->CheckSdmaReaches(remoteRank);
    }

    return false;
}

hybm_data_op_type MemEntityDefault::CanReachDataOperators(uint32_t remoteRank) const noexcept
{
    uint32_t supportDataOp = 0U;
    bool sdmaReach = SdmaReaches(remoteRank);   // SDMA reaches mean MTE reaches too
    if (sdmaReach) {
        supportDataOp |= HYBM_DOP_TYPE_MTE;
    }
    if (sdmaReach && ((options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_SDMA) != 0)) {
        supportDataOp |= HYBM_DOP_TYPE_DEVICE_SDMA;
    }
    if ((options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_RDMA) != 0) {
        supportDataOp |= HYBM_DOP_TYPE_DEVICE_RDMA;
    }
    if (sdmaReach && ((options_.bmDataOpType & HYBM_DOP_TYPE_DEVICE_UDMA) != 0)) {
        supportDataOp |= HYBM_DOP_TYPE_DEVICE_UDMA;
    }

    return static_cast<hybm_data_op_type>(supportDataOp);
}

void *MemEntityDefault::GetReservedMemoryPtr(hybm_mem_type memType) noexcept
{
    if (memType == HYBM_MEM_TYPE_DEVICE) {
        return hbmGva_;
    }

    if (memType == HYBM_MEM_TYPE_HOST) {
        return dramGva_;
    }

    return nullptr;
}

void MemEntityDefault::ReleaseResources()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized) {
        return;
    }
    if (transportManager_) {
        transportManager_->CloseDevice();
        transportManager_ = nullptr;
    }
    hbmSegment_.reset();
    dramSegment_.reset();
    initialized = false;
}

}  // namespace shm
