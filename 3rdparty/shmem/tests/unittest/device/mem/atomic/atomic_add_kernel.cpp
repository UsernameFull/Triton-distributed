/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

#include "shmem.h"
#include "unittest/utils/func_type.h"
constexpr uint64_t MESSAGE_SIZE = 64;

#define ATOMIC_ADD_TEST_KERNEL(NAME, TYPE)                                                              \
    extern "C" __global__ __aicore__ void test_atomic_add_##NAME##_kernel(GM_ADDR gva, uint64_t config) \
    {                                                                                                   \
        util_set_ffts_config(config);                                                                   \
        int64_t rank = aclshmem_my_pe();                                                                \
        int64_t rank_size = aclshmem_n_pes();                                                           \
        GM_ADDR dst_addr;                                                                               \
                                                                                                        \
        for (int64_t peer = 0; peer < rank_size; peer++) {                                              \
            if (peer == rank) {                                                                         \
                continue;                                                                               \
            }                                                                                           \
            dst_addr = gva + rank * MESSAGE_SIZE;                                                       \
            if (AscendC::GetSubBlockIdx() == 0) {                                                       \
                aclshmem_##NAME##_atomic_add((__gm__ TYPE*)dst_addr, 1, peer);                          \
            }                                                                                           \
        }                                                                                               \
        aclshmem_barrier_all();                                                                         \
    }
ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_KERNEL(ATOMIC_ADD_TEST_KERNEL);

#define ATOMIC_ADD_TEST(NAME, TYPE)                                                                   \
    void test_atomic_add_##NAME##_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config) \
    {                                                                                                 \
        test_atomic_add_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                 \
    }
ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_KERNEL(ATOMIC_ADD_TEST);
