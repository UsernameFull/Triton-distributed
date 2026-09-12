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
#include "unittest_main_test.h"

static constexpr int kInputLength = 16;
static constexpr int kTestOffset  = 10;

#define TEST_FUNC_BLOCKING(NAME, TYPE) \
    extern void test_##NAME##_put_blocking(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, uint8_t *dev_ptr); \
    extern void test_##NAME##_get_blocking(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, uint8_t *dev_ptr)

ACLSHMEM_FUNC_TYPE_HOST(TEST_FUNC_BLOCKING);

#define TEST_PUT_GET_BLOCKING(NAME, TYPE)                                                                               \
    static void test_##NAME##_put_get_blocking(aclrtStream stream, uint8_t *gva, uint32_t pe_id, uint32_t pe_size)      \
    {                                                                                                                   \
        const int total_size = kInputLength * (int)pe_size;                                                             \
        const size_t input_size = (size_t)total_size * sizeof(TYPE);                                                    \
                                                                                                                        \
        std::vector<TYPE> input(total_size, (TYPE)0);                                                                   \
        for (int i = 0; i < kInputLength; i++) {                                                                        \
            input[i] = static_cast<TYPE>(pe_id + kTestOffset);                                                          \
        }                                                                                                               \
                                                                                                                        \
        void *dev_ptr = nullptr;                                                                                        \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                    \
        ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);            \
                                                                                                                        \
        uint32_t block_dim = 1;                                                                                         \
        void *ptr = aclshmem_malloc(input_size);                                                                        \
        ASSERT_NE(ptr, nullptr);                                                                                        \
                                                                                                                        \
        test_##NAME##_put_blocking(block_dim, stream, util_get_ffts_config(), (uint8_t *)ptr, (uint8_t *)dev_ptr);      \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                   \
                                                                                                                        \
        test_##NAME##_get_blocking(block_dim, stream, util_get_ffts_config(), (uint8_t *)ptr, (uint8_t *)dev_ptr);      \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                   \
                                                                                                                        \
        ASSERT_EQ(aclrtMemcpy(input.data(), input_size, dev_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);            \
                                                                                                                        \
        int32_t flag = 0;                                                                                               \
        for (int i = 0; i < total_size; i++) {                                                                          \
            int stage = i / kInputLength;                                                                               \
            int src_pe = (stage - 1 + (int)pe_size) % (int)pe_size;                                                     \
            if (input[i] != static_cast<TYPE>(src_pe + kTestOffset)) {                                                  \
                flag = 1;                                                                                               \
                break;                                                                                                  \
            }                                                                                                           \
        }                                                                                                               \
        ASSERT_EQ(flag, 0);                                                                                             \
                                                                                                                        \
        aclshmem_free(ptr);                                                                                             \
        ASSERT_EQ(aclrtFree(dev_ptr), 0);                                                                               \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_PUT_GET_BLOCKING);

#define TEST_ACLSHMEM_MEM_BLOCKING(NAME, TYPE)                                                  \
    void test_##NAME##_aclshmem_mem_blocking(int pe_id, int n_pes, uint64_t local_mem_size)     \
    {                                                                                           \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                             \
        aclrtStream stream;                                                                     \
        test_init(pe_id, n_pes, local_mem_size, &stream);                                       \
        ASSERT_NE(stream, nullptr);                                                             \
                                                                                                \
        test_##NAME##_put_get_blocking(stream, (uint8_t *)g_state.heap_base, pe_id, n_pes);     \
        std::cout << "[TEST] begin to exit...... pe_id: " << pe_id << std::endl;                \
        test_finalize(stream, device_id);                                                       \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_ACLSHMEM_MEM_BLOCKING);

#define TESTAPI_BLOCKING(NAME, TYPE)                                                        \
    TEST(TestMemApi, TestShmemGM##NAME##MemBlocking)                                        \
    {                                                                                       \
        const int process_count = test_gnpu_num;                                            \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                   \
        test_mutil_task(test_##NAME##_aclshmem_mem_blocking, local_mem_size, process_count);\
    }

ACLSHMEM_FUNC_TYPE_HOST(TESTAPI_BLOCKING);
