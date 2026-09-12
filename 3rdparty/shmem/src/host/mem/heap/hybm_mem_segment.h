/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBM_MEM_SEGMENT_H
#define MEM_FABRIC_HYBRID_HYBM_MEM_SEGMENT_H

#include <memory>
#include <vector>
#include "hybm_mem_common.h"
#include "hybm_mem_slice.h"
#include "transport_def.h"
#include "dl_comm_def.h"

namespace shm {
struct TransportExtraInfo {
    uint32_t rankId;
    transport::TransportMemoryKey memKey;
};

class MemSegment;
using MemSegmentPtr = std::shared_ptr<MemSegment>;

struct MemSliceStatus {
    std::shared_ptr<MemSlice> slice;
    void *handle;

    explicit MemSliceStatus(std::shared_ptr<MemSlice> s) noexcept : slice{std::move(s)}, handle(nullptr) {}
    MemSliceStatus(std::shared_ptr<MemSlice> s, void *h) noexcept : slice{std::move(s)}, handle(h) {}
};

class MemSegment {
public:
    static MemSegmentPtr Create(const MemSegmentOptions &options, int entityId);

public:
    explicit MemSegment(const MemSegmentOptions &options, int eid) : options_{options}, entityId_{eid} {}
    virtual ~MemSegment() = default;

    /*
     * Validate options
     * @return 0 if ok
     */
    virtual Result ValidateOptions() noexcept = 0;

    virtual Result ReserveMemorySpace(void **address) noexcept = 0;

    virtual Result UnReserveMemorySpace() noexcept = 0;

    /*
     * Allocate memory according to segType
     * @return 0 if successful
     */
    virtual Result AllocLocalMemory(uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept = 0;

    /*
     * register memory according to segType
     * @return 0 if successful
     */
    virtual Result RegisterMemory(const void *addr, uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept = 0;

    /*
     * release one slice
     * @return 0 if successful
     */
    virtual Result ReleaseSliceMemory(const std::shared_ptr<MemSlice> &slice) noexcept = 0;

    /*
     * Export exchange info according to infoExType
     * @return exchange info
     */
    virtual Result Export(std::string &exInfo) noexcept = 0;

    virtual Result Export(const std::shared_ptr<MemSlice> &slice, std::string &exInfo) noexcept = 0;

    virtual Result GetExportSliceSize(size_t &size) noexcept = 0;

    /*
     * Import exchange info and translate it into data structure
     * @param allExInfo
     */
    virtual Result Import(const std::vector<std::string> &allExInfo, void *addresses[]) noexcept = 0;

    /*
     * delete imported memory area according to rankid
     * @return 0 if successful
     */
    virtual Result RemoveImported(const std::vector<uint32_t>& ranks) noexcept = 0;

    /*
     * @brief after Import exchange info, should call this func to make it work
     * @return 0 if successful
     */
    virtual Result Mmap() noexcept = 0;

    /*
     * After remove imported exchange info, should call this func to make it work
     * @return 0 if successful
     */
    virtual Result Unmap() noexcept = 0;

    virtual std::shared_ptr<MemSlice> GetMemSlice(hybm_mem_slice_t slice) const noexcept = 0;

    /*
     * check memery area in this segment
     * @return true if in range
     */
    virtual bool MemoryInRange(const void *begin, uint64_t size) const noexcept = 0;

    /*
     * check memery area in this segment
     * @return true if in range
    */
    virtual bool GetRankIdByAddr(const void *addr, uint64_t size, uint32_t &rankId) const noexcept = 0;

    /*
     * get memory type
     */
    virtual hybm_mem_type GetMemoryType() const noexcept = 0;

    virtual bool CheckSdmaReaches(uint32_t rankId) const noexcept;

protected:
    static Result InitDeviceInfo();

    static void FillSysBootIdInfo() noexcept;

    /**
     * check whether in the same node
     */
    static bool CanLocalHostReaches(uint32_t superPodId, uint32_t serverId, uint32_t deviceId) noexcept;

    /**
     * check whether the SDMA accessible to the other device
     */
    static bool IsSdmaAccessible(uint32_t superPodId, uint32_t serverId, uint32_t deviceId) noexcept;

protected:
    const MemSegmentOptions options_;
    const int entityId_;

    static bool deviceInfoReady;
    static int deviceId_;
    static int logicDeviceId_;
    static uint32_t pid_;
    static uint32_t sdid_;
    static uint32_t serverId_;
    static uint32_t superPodId_;
    static AscendSocType socType_;
    static std::string sysBoolId_;
    static uint32_t bootIdHead_;
};
}

#endif // MEM_FABRIC_HYBRID_HYBM_MEM_SEGMENT_H
