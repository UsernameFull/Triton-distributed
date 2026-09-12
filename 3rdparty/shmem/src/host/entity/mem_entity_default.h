/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBM_ENGINE_IMPL_H
#define MEM_FABRIC_HYBRID_HYBM_ENGINE_IMPL_H

#include <map>
#include <mutex>
#include "hybm_device_mem_segment.h"
#include "hybm_mem_segment.h"
#include "mem_entity_base.h"
#include "hybm_mem_common.h"
#include "transport_manager.h"
#include "mem_entity_def.h"

namespace shm {
struct EntityExportInfo {
    uint64_t magic{ENTITY_EXPORT_INFO_MAGIC};
    uint64_t version{EXPORT_INFO_VERSION};
    uint16_t rankId{0};
    uint16_t role{0};
    uint32_t reserved{0};
    char nic[64]{};
};
struct SliceExportTransportKey {
    uint64_t magic;
    uint16_t rankId;
    uint16_t reserved[3]{};
    uint64_t address;
    transport::TransportMemoryKey key;
    SliceExportTransportKey() : SliceExportTransportKey{0, 0, 0} {}
    SliceExportTransportKey(uint64_t mag, uint16_t rank, uint64_t addr)
        : magic{mag}, rankId{rank}, address{addr}, key{0}
    {}
};

class MemEntityDefault : public MemEntity {
public:
    explicit MemEntityDefault(int32_t id) noexcept;
    ~MemEntityDefault() override;

    int32_t Initialize(const hybm_options *options) noexcept override;
    void UnInitialize() noexcept override;

    int32_t ReserveMemorySpace(void **reservedMem) noexcept override;
    int32_t UnReserveMemorySpace() noexcept override;
    void *GetReservedMemoryPtr(hybm_mem_type memType) noexcept override;

    int32_t AllocLocalMemory(uint64_t size, hybm_mem_type mType, uint32_t flags, hybm_mem_slice_t &slice) noexcept override;
    int32_t RegisterLocalMemory(const void *ptr, uint64_t size, uint32_t flags,
                                hybm_mem_slice_t &slice) noexcept override;
    int32_t FreeLocalMemory(hybm_mem_slice_t slice, uint32_t flags) noexcept override;

    int32_t ExportExchangeInfo(ExchangeInfoWriter &desc, uint32_t flags) noexcept override;
    int32_t ExportExchangeInfo(hybm_mem_slice_t slice, ExchangeInfoWriter &desc, uint32_t flags) noexcept override;
    int32_t ImportExchangeInfo(const ExchangeInfoReader *desc, uint32_t count, void *addresses[],
                               uint32_t flags) noexcept override;
    int32_t RemoveImported(const std::vector<uint32_t>& ranks) noexcept override;
    int32_t GetExportSliceInfoSize(size_t &size) noexcept override;

    int32_t SetExtraContext(const void *context, uint32_t size) noexcept override;

    int32_t Mmap() noexcept override;
    void Unmap() noexcept override;

    bool CheckAddressInEntity(const void *ptr, uint64_t length) const noexcept override;
    bool SdmaReaches(uint32_t remoteRank) const noexcept override;
    hybm_data_op_type CanReachDataOperators(uint32_t remoteRank) const noexcept override;

private:
    static int CheckOptions(const hybm_options *options) noexcept;
    int LoadExtendLibrary() noexcept;
    int UpdateHybmDeviceInfo(uint32_t extCtxSize) noexcept;
    void SetHybmDeviceInfo(HybmDeviceMeta &info);

    int32_t ExportWithSlice(hybm_mem_slice_t slice, ExchangeInfoWriter &desc, uint32_t flags) noexcept;
    int32_t ExportWithoutSlice(ExchangeInfoWriter &desc, uint32_t flags) noexcept;
    int32_t ImportForTransport(const ExchangeInfoReader desc[], uint32_t count, uint32_t flags) noexcept;
    int32_t ImportForTransportPrecheck(const ExchangeInfoReader desc[], uint32_t &cnt, bool &entity);

    Result InitSegment();
    Result InitHbmSegment();
    Result InitDramSegment();
    Result InitTransManager();

    void ReleaseResources();
    int32_t SetThreadAclDevice();

private:
    bool initialized = false;
    std::mutex mutex_;
    const int32_t id_; /* id of the engine */
    static thread_local bool isSetDevice_;
    hybm_options options_{};
    void *hbmGva_{nullptr};
    void *dramGva_{nullptr};
    std::shared_ptr<MemSegment> hbmSegment_{nullptr};
    std::shared_ptr<MemSegment> dramSegment_{nullptr};
    transport::TransManagerPtr transportManager_;
    std::mutex importMutex_;
    std::unordered_map<uint32_t, EntityExportInfo> importedRanks_;
    std::unordered_map<uint32_t, std::unordered_map<uint64_t, transport::TransportMemoryKey>> importedMemories_;
};
using EngineImplPtr = std::shared_ptr<MemEntityDefault>;
}

#endif  // MEM_FABRIC_HYBRID_HYBM_ENGINE_IMPL_H
