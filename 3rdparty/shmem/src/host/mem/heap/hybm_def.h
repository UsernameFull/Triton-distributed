/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBRID_BIG_MEM_DL_H
#define MEM_FABRIC_HYBRID_HYBRID_BIG_MEM_DL_H

#include <stdint.h>

typedef void *hybm_entity_t;
typedef void *hybm_mem_slice_t;

constexpr uint64_t KB = 1024ULL;
constexpr uint64_t MB = KB * 1024ULL;
constexpr uint64_t GB = MB * 1024ULL;
constexpr uint64_t TB = GB * 1024ULL;

#define HYBM_FREE_SINGLE_SLICE 0x00
#define HYBM_FREE_ALL_SLICE 0x01

#define HYBM_EXPORT_PARTIAL_SLICE 0x00
#define HYBM_EXPORT_ALL_SLICE 0x01

#define HYBM_PERFORMANCE_MODE_FLAG_INDEX 7
#define HYBM_PERFORMANCE_MODE_FLAG_LEN   1
#define HYBM_BIND_NUMA_FLAG_INDEX        0
#define HYBM_BIND_NUMA_FLAG_LEN          7

typedef enum {
    HYBM_TYPE_AI_CORE_INITIATE = 0,
    HYBM_TYPE_BUTT
} hybm_type;

typedef enum {
    HYBM_DOP_TYPE_DEFAULT = 0U,
    HYBM_DOP_TYPE_MTE = 1U << 0,
    HYBM_DOP_TYPE_DEVICE_RDMA = 1U << 1,
    HYBM_DOP_TYPE_DEVICE_SDMA = 1U << 2,
    HYBM_DOP_TYPE_DEVICE_UDMA = 1U << 3,
    HYBM_DOP_TYPE_BUTT
} hybm_data_op_type;

typedef enum {
    HYBM_SCOPE_IN_NODE = 0,
    HYBM_SCOPE_CROSS_NODE,

    HYBM_SCOPE_BUTT
} hybm_scope;

typedef enum {
    HYBM_MEM_TYPE_DEVICE = 1U << 0,
    HYBM_MEM_TYPE_HOST = 1U << 1,
    HYBM_MEM_TYPE_BUTT
} hybm_mem_type;

typedef enum {
    HYBM_ROLE_PEER = 0,
    HYBM_ROLE_SENDER,
    HYBM_ROLE_RECEIVER,
    HYBM_ROLE_BUTT
} hybm_role_type;

typedef struct {
    uint8_t desc[512L];
    uint32_t descLen;
} hybm_exchange_info;

typedef struct {
    hybm_type bmType;
    hybm_mem_type memType;
    hybm_data_op_type bmDataOpType;
    hybm_scope bmScope;
    uint16_t rankCount;
    uint16_t rankId;
    uint16_t devId;
    uint64_t deviceVASpace;
    uint64_t hostVASpace;
    uint64_t preferredGVA;
    hybm_role_type role;
    uint32_t flags;
    char nic[64];
} hybm_options;

#endif // MEM_FABRIC_HYBRID_HYBRID_BIG_MEM_DL_H
