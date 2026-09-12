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
#include "../utils/func_type.h"
#include "unittest_main_test.h"

const int test_mul = 10;

#define TEST_FUNC(NAME, TYPE)                                                                           \
    extern void test_ub_##NAME##_put(uint32_t block_dim, void *stream, uint8_t *gva, uint8_t *dev_ptr); \
    extern void test_ub_##NAME##_get(uint32_t block_dim, void *stream, uint8_t *gva, uint8_t *dev_ptr)

ACLSHMEM_FUNC_TYPE_HOST(TEST_FUNC);

#define TEST_UB_PUT_GET(NAME, TYPE)                                                                              \
    static void test_ub_##NAME##_put_get(aclrtStream stream, uint8_t *gva, uint32_t rank_id, uint32_t rank_size) \
    {                                                                                                            \
        int total_size = 512;                                                                                    \
        size_t input_size = total_size * sizeof(TYPE);                                                           \
                                                                                                                 \
        std::vector<TYPE> input(total_size, 0);                                                                  \
        for (int i = 0; i < total_size; i++) {                                                                   \
            input[i] = static_cast<TYPE>(rank_id * test_mul);                                                    \
        }                                                                                                        \
                                                                                                                 \
        void *dev_ptr;                                                                                           \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                             \
                                                                                                                 \
        ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);     \
                                                                                                                 \
        uint32_t block_dim = 1;                                                                                  \
        void *ptr = aclshmem_malloc(total_size * sizeof(TYPE));                                                  \
        test_ub_##NAME##_put(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                            \
                                                                                                                 \
        ASSERT_EQ(aclrtMemcpy(input.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);         \
                                                                                                                 \
        test_ub_##NAME##_get(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                            \
                                                                                                                 \
        ASSERT_EQ(aclrtMemcpy(input.data(), input_size, dev_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);     \
                                                                                                                 \
        /* result check */                                                                                       \
        int32_t flag = 0;                                                                                        \
        for (int i = 0; i < total_size; i++) {                                                                   \
            int golden = rank_id % rank_size;                                                                    \
            if (input[i] != static_cast<TYPE>(golden * test_mul))                                                \
                flag = 1;                                                                                        \
        }                                                                                                        \
        ASSERT_EQ(flag, 0);                                                                                      \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_UB_PUT_GET);

#define TEST_ACLSHMEM_UB_MEM(NAME, TYPE)                                                       \
    void test_aclshmem_ub_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)      \
    {                                                                                          \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                          \
        aclrtStream stream;                                                                    \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                  \
        ASSERT_NE(stream, nullptr);                                                            \
                                                                                               \
        test_ub_##NAME##_put_get(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);      \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;           \
        test_finalize(stream, device_id);                                                      \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_ACLSHMEM_UB_MEM);

#define TESTAPI(NAME, TYPE)                                                             \
    TEST(TestMemApi, TestShmemUB##NAME##Mem)                                            \
    {                                                                                   \
        const int processCount = test_gnpu_num;                                         \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                               \
        test_mutil_task(test_aclshmem_ub_##NAME##_mem, local_mem_size, processCount);   \
    }

ACLSHMEM_FUNC_TYPE_HOST(TESTAPI);
