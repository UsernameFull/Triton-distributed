/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBM_MM_COMMON_H
#define MEM_FABRIC_HYBRID_HYBM_MM_COMMON_H

#include <memory>
#include "hybm_def.h"

namespace shm {

constexpr auto DEVICE_SHM_NAME_SIZE = 64U;

constexpr uint64_t EXPORT_INFO_MAGIC = 0xAABB1234FFFFEEEEUL;
constexpr uint64_t EXPORT_SLICE_MAGIC = 0xAABB1234FFFFBBBBUL;
constexpr uint64_t ENTITY_EXPORT_INFO_MAGIC = 0xAABB1234FFFFEE00UL;
constexpr uint64_t HBM_SLICE_EXPORT_INFO_MAGIC = 0xAABB1234FFFFEE01UL;
constexpr uint64_t DRAM_SLICE_EXPORT_INFO_MAGIC = 0xAABB1234FFFFEE02UL;
constexpr uint64_t EXPORT_INFO_VERSION = 0x1UL;

class MemSlice;
class MemSegment;

using MemSlicePtr = std::shared_ptr<MemSlice>;
using MemSegmentPtr = std::shared_ptr<MemSegment>;

enum MemType : uint8_t {
    MEM_TYPE_HOST_DRAM = 0,
    MEM_TYPE_DEVICE_HBM,

    MEM_TYPE_DEVICE_BUTT
};

enum MemPageTblType : uint8_t {
    MEM_PT_TYPE_SVM = 0,
    MEM_PT_TYPE_GVM,
    MEM_PT_TYPE_HYM,

    MEM_PT_TYPE_BUTT
};

enum MemAddrType : uint8_t {
    MEM_ADDR_TYPE_VIRTUAL = 0,
    MEM_ADDR_TYPE_PHYSICAL,

    MEM_ADDR_TYPE_BUTT
};

enum MemSegType : uint8_t {
    HYBM_MST_HBM = 0,
    HYBM_MST_DRAM,

    HYBM_MST_BUTT
};

enum MemSegInfoExchangeType : uint8_t {
    HYBM_INFO_EXG_IN_NODE,
    HYBM_INFO_EXG_CROSS_NODE_HCCS,
    HYBM_INFO_EXG_CROSS_NODE_SDMA,
    HYBM_INFO_EXG_CROSS_NODE_UDMA,

    HYBM_INFO_EXG_BUTT
};

struct MemSegmentOptions {
    int32_t devId = 0;
    hybm_role_type role = HYBM_ROLE_PEER;
    hybm_data_op_type dataOpType = HYBM_DOP_TYPE_MTE;
    MemSegType segType = HYBM_MST_HBM;
    MemSegInfoExchangeType infoExType = HYBM_INFO_EXG_IN_NODE;
    bool shared = true;
    uint64_t size = 0;
    uint32_t rankId = 0;   // must start from 0 and increase continuously
    uint32_t rankCnt = 0;  // total rank count
    uint32_t flags = 0;
};

}  // namespace shm

#endif  // MEM_FABRIC_HYBRID_HYBM_MM_COMMON_H
