/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INFO_H
#define INFO_H

#include <limits.h>

static uint64_t ACLSHMEM_MALLOC_MAX_SIZE = 1024UL * 1024UL * 1024;
constexpr uint32_t M0 = 128;
constexpr uint32_t N0 = 256;
constexpr uint32_t K0 = 256;
constexpr uint32_t WORKSPACE_STAGES = 2;
constexpr uint32_t UB_STAGES = 2;
constexpr uint32_t BLOCK_NUM = 20;
constexpr uint32_t WARM_UP_TIMES = 10;
constexpr uint32_t PERF_TEST_CYCLE_TIMES = 3;
constexpr uint32_t MAX_BLOCK_COUNT = 2;
constexpr int32_t ACLSHMEM_BUFF_BYTES = 256UL * 1024 * 1024;
constexpr int32_t FLAG_BUFF_BYTES = 5 * 512 * 1024;
constexpr int32_t INPUT_DTYPE = 2;

struct CocTilingParams {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t commTileM = 0;
    uint32_t commInterval = 0;
    uint32_t commNpuSplit = 0;
    uint32_t commDataSplit = 0;
    uint32_t commBlockM = 0;
    uint32_t rankSize = 0;
};

struct COCMatMulInfo {
    int64_t m;
    int64_t k;
    int64_t n;
};

#endif // INFO_H