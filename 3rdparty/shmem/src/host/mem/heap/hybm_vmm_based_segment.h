/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MF_HYBRID_HYBM_VMM_BASED_SEGMENT_H
#define MF_HYBRID_HYBM_VMM_BASED_SEGMENT_H

#include "hybm_mem_segment.h"
#include "dl_hal_api.h"

namespace shm {
struct MemExportInfo {
    uint64_t magic{HBM_SLICE_EXPORT_INFO_MAGIC};
    uint64_t version{EXPORT_INFO_VERSION};
    uint64_t mappingOffset{0};
    uint32_t sliceIndex{0};
    uint32_t sdid{0};
    int32_t logicDeviceId{-1};
    int32_t deviceId{-1};
    uint32_t serverId{0};
    uint32_t superPodId{0};
    uint32_t rankId{0};
    uint64_t size{0};
    MemShareHandle shareHandle;
};

class HybmVmmBasedSegment : public MemSegment {
public:
    explicit HybmVmmBasedSegment(const MemSegmentOptions &options, int eid) : MemSegment{options, eid} {}

    ~HybmVmmBasedSegment() override {}

    Result ValidateOptions() noexcept override;
    Result ReserveMemorySpace(void **address) noexcept override;
    Result UnReserveMemorySpace() noexcept override;
    Result AllocLocalMemory(uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept override;
    Result RegisterMemory(const void *addr, uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept override;
    Result ReleaseSliceMemory(const std::shared_ptr<MemSlice> &slice) noexcept override;
    Result Export(std::string &exInfo) noexcept override;
    Result Export(const std::shared_ptr<MemSlice> &slice, std::string &exInfo) noexcept override;
    Result GetExportSliceSize(size_t &size) noexcept override;
    Result Import(const std::vector<std::string> &allExInfo, void *addresses[]) noexcept override;
    Result RemoveImported(const std::vector<uint32_t> &ranks) noexcept override;
    Result Mmap() noexcept override;
    Result Unmap() noexcept override;
    std::shared_ptr<MemSlice> GetMemSlice(hybm_mem_slice_t slice) const noexcept override;
    bool MemoryInRange(const void *begin, uint64_t size) const noexcept override;
    bool GetRankIdByAddr(const void *addr, uint64_t size, uint32_t &rankId) const noexcept override;

    hybm_mem_type GetMemoryType() const noexcept override
    {
        return HYBM_MEM_TYPE_DEVICE;
    }
    bool CheckSdmaReaches(uint32_t rankId) const noexcept override;
    static bool CanMapRemote(const MemExportInfo &rmi) noexcept;
    static void GetDeviceInfo(uint32_t &sdId, uint32_t &serverId, uint32_t &superPodId) noexcept;

private:
    Result ExportInner(const std::shared_ptr<MemSlice> &slice, MemShareHandle &sHandle) noexcept;

    std::vector<MemExportInfo> imports_;
    uint8_t *globalVirtualAddress_{nullptr};
    std::vector<uint64_t> reservedVirtualAddresses_;
    uint64_t totalVirtualSize_{0UL};
    uint64_t allocatedSize_{0UL};
    uint16_t sliceCount_{0};

    std::map<uint16_t, MemSliceStatus> slices_;
    std::map<uint16_t, std::string> exportMap_;
    std::map<uint64_t, drv_mem_handle_t *> mappedMem_;
    std::map<uint16_t, MemExportInfo> importMap_;
};
} // namespace shm

#endif // MF_HYBRID_HYBM_VMM_BASED_SEGMENT_H