/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "opdev/bfloat16.h"
#include "opdev/fp16_t.h"
#include "func_type.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream* st);
extern void test_finalize(aclrtStream stream, int device_id);

#define TEST_ATOMIC_ADD_FUNC(NAME, TYPE) \
    extern void test_atomic_add_##NAME##_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_HOST(TEST_ATOMIC_ADD_FUNC);

#define TEST_ACLSHMEM_ATOMIC_ADD_HOST(NAME, TYPE)                                                                     \
    static void test_atomic_add_##NAME##_host(aclrtStream stream, uint8_t* gva, uint32_t rank_id, uint32_t rank_size) \
    {                                                                                                                 \
        size_t messageSize = 64;                                                                                      \
        TYPE* xHost;                                                                                                  \
        size_t totalSize = messageSize * rank_size;                                                                   \
                                                                                                                      \
        ASSERT_EQ(aclrtMallocHost((void**)(&xHost), totalSize), 0);                                                   \
        for (uint32_t i = 0; i < messageSize / sizeof(TYPE); i++) {                                                   \
            xHost[i] = rank_id + 1;                                                                                   \
        }                                                                                                             \
                                                                                                                      \
        uint8_t* ptr = (uint8_t*)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(                                                                                                    \
            aclrtMemcpy(ptr + rank_id * messageSize, messageSize, xHost, messageSize, ACL_MEMCPY_HOST_TO_DEVICE), 0); \
                                                                                                                      \
        uint32_t block_dim = 3;                                                                                       \
        test_atomic_add_##NAME##_do(block_dim, stream, (uint8_t*)ptr, util_get_ffts_config());                        \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                 \
                                                                                                                      \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                            \
        std::cout << p_name;                                                                                          \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                       \
        for (uint32_t i = 0; i < rank_size; i++) {                                                                    \
            if (i == rank_id) {                                                                                       \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(rank_id + 1));                     \
            } else {                                                                                                  \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(block_dim));                       \
            }                                                                                                         \
        }                                                                                                             \
    }
ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_HOST(TEST_ACLSHMEM_ATOMIC_ADD_HOST);

#define TEST_CLEANUP_AND_EXIT(rank_id, stream, device_id)                              \
    do {                                                                               \
        std::cout << "[TEST] begin to exit...... rank_id: " << (rank_id) << std::endl; \
        test_finalize(stream, device_id);                                              \
    } while (0)

#define TEST_ACLSHMEM_ATOMIC_ADD(NAME, TYPE)                                                      \
    void test_aclshmem_atomic_add_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size) \
    {                                                                                             \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                             \
        aclrtStream stream;                                                                       \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                     \
        ASSERT_NE(stream, nullptr);                                                               \
        test_atomic_add_##NAME##_host(stream, (uint8_t*)g_state.heap_base, rank_id, n_ranks);     \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;              \
        test_finalize(stream, device_id);                                                         \
    }
ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_HOST(TEST_ACLSHMEM_ATOMIC_ADD);

#define TEST_ATOMIC_ADD_API(NAME, TYPE)                                                       \
    TEST(TestMemApi, TestShmemAtomicAdd##NAME##Mem)                                           \
    {                                                                                         \
        const int processCount = test_gnpu_num;                                               \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                       \
        test_mutil_task(test_aclshmem_atomic_add_##NAME##_mem, local_mem_size, processCount); \
    }

ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_HOST(TEST_ATOMIC_ADD_API);
