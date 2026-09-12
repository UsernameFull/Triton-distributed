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

#define TEST_FUNC(NAME, TYPE)                                                                                        \
    extern void test_ub_##NAME##_non_contiguous_put(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, \
                                                    uint8_t *dev_ptr, int repeat, int length);                       \
    extern void test_ub_##NAME##_non_contiguous_get(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, \
                                                    uint8_t *dev_ptr, int repeat, int length)

ACLSHMEM_FUNC_TYPE_HOST(TEST_FUNC);

constexpr int input_repeat = 32;
constexpr int input_length = 32;

#define TEST_UB_NON_CONTIGUOUS_PUT_GET(NAME, TYPE)                                                           \
    static void TestUB##NAME##NonContiguousPutGet(aclrtStream stream, uint8_t *gva, uint32_t rank_id,        \
                                                  uint32_t rank_size)                                        \
    {                                                                                                        \
        int rank_flag = rank_id * 10; /* Used as input data */                                               \
        int total_size = input_repeat * input_length;                                                        \
        size_t input_size = total_size * sizeof(TYPE);                                                       \
                                                                                                             \
        std::vector<TYPE> input(total_size, 0);                                                              \
        for (int i = 0; i < input_repeat; i++) {                                                             \
            for (int j = 0; j < input_length; j++) {                                                         \
                input[i * input_length + j] = static_cast<TYPE>(rank_flag) + static_cast<TYPE>(i);           \
            }                                                                                                \
        }                                                                                                    \
                                                                                                             \
        void *dev_ptr;                                                                                       \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                         \
        ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0); \
                                                                                                             \
        uint32_t block_dim = 1;                                                                              \
        void *ptr = aclshmem_malloc(total_size * sizeof(TYPE));                                              \
        test_ub_##NAME##_non_contiguous_put(block_dim, stream, util_get_ffts_config(), (uint8_t *)ptr,       \
                                            (uint8_t *)dev_ptr, input_repeat, input_length);                 \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                        \
                                                                                                             \
        ASSERT_EQ(aclrtMemcpy(input.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);     \
        test_ub_##NAME##_non_contiguous_get(block_dim, stream, util_get_ffts_config(), (uint8_t *)ptr,       \
                                            (uint8_t *)dev_ptr, input_repeat / 2, input_length);             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                        \
                                                                                                             \
        ASSERT_EQ(aclrtMemcpy(input.data(), input_size, dev_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0); \
        /* result check */                                                                                   \
        int32_t flag = 0;                                                                                    \
        for (int i = 0; i < input_repeat / 4; i++) {                                                         \
            for (int j = 0; j < input_length; j++) {                                                         \
                int golden = rank_id % rank_size;                                                            \
                if (input[i * input_length + j] != static_cast<TYPE>(rank_flag) + static_cast<TYPE>(i * 4))  \
                    flag = 1;                                                                                \
            }                                                                                                \
        }                                                                                                    \
        ASSERT_EQ(flag, 0);                                                                                  \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_UB_NON_CONTIGUOUS_PUT_GET);

#define TEST_ACLSHMEM_UB_NON_CONTIGUOUS(NAME, TYPE)                                                     \
    void test_aclshmem_ub_##NAME##_non_contiguous(int rank_id, int n_ranks, uint64_t local_mem_size)    \
    {                                                                                                   \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                   \
        aclrtStream stream;                                                                             \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                           \
        ASSERT_NE(stream, nullptr);                                                                     \
                                                                                                        \
        TestUB##NAME##NonContiguousPutGet(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);      \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                    \
        test_finalize(stream, device_id);                                                               \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_ACLSHMEM_UB_NON_CONTIGUOUS)

#define TESTAPI(NAME, TYPE)                                                                      \
    TEST(TestMemApi, TestShmemUB##NAME##NonContiguous)                                           \
    {                                                                                            \
        const int processCount = test_gnpu_num;                                                  \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                        \
        test_mutil_task(test_aclshmem_ub_##NAME##_non_contiguous, local_mem_size, processCount); \
    }

ACLSHMEM_FUNC_TYPE_HOST(TESTAPI)
