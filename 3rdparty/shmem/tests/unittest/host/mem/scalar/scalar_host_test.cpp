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
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem.h"
#include "opdev/bfloat16.h"
#include "opdev/fp16_t.h"
#include "../utils/func_type.h"
#include "unittest_main_test.h"

const size_t total_size = 1024;
const float test_offset = 3.5f;

#define PUT_ONE_NUM_DO(NAME, TYPE)                                                                       \
    extern void put_##NAME##_one_num_do(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, \
                                        uint8_t *dev);                                                   \
    extern void get_##NAME##_one_num_do(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, uint8_t *dev)

ACLSHMEM_FUNC_TYPE_HOST(PUT_ONE_NUM_DO);

#define TEST_SCALAR_PUT_GET(NAME, TYPE)                                                                            \
    static int32_t test_##NAME##_scalar_put_get(aclrtStream stream, uint32_t rank_id, uint32_t rank_size)          \
    {                                                                                                              \
        TYPE *y_host;                                                                                              \
        size_t input_size = total_size * sizeof(TYPE);                                                             \
        EXPECT_EQ(aclrtMallocHost((void **)(&y_host), input_size), 0); /* size = 1024 */                           \
                                                                                                                   \
        void *dev_ptr;                                                                                             \
        EXPECT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                               \
                                                                                                                   \
        uint32_t block_dim = 1;                                                                                    \
                                                                                                                   \
        TYPE value = static_cast<TYPE>(test_offset) + (TYPE)rank_id;                                               \
        EXPECT_EQ(aclrtMemcpy(dev_ptr, 1 * sizeof(TYPE), &value, 1 * sizeof(TYPE), ACL_MEMCPY_DEVICE_TO_HOST), 0); \
        void *ptr = aclshmem_malloc(total_size);                                                                   \
        put_##NAME##_one_num_do(block_dim, stream, util_get_ffts_config(), (uint8_t *)ptr, (uint8_t *)dev_ptr);    \
        EXPECT_EQ(aclrtSynchronizeStream(stream), 0);                                                              \
                                                                                                                   \
        EXPECT_EQ(aclrtMemcpy(y_host, 1 * sizeof(TYPE), ptr, 1 * sizeof(TYPE), ACL_MEMCPY_DEVICE_TO_HOST), 0);     \
                                                                                                                   \
        /* result check */                                                                                         \
        int32_t flag = 0;                                                                                          \
        if (y_host[0] != static_cast<TYPE>(test_offset + (rank_id + rank_size - 1) % rank_size))                   \
            flag = 1;                                                                                              \
                                                                                                                   \
        get_##NAME##_one_num_do(block_dim, stream, util_get_ffts_config(), (uint8_t *)ptr, (uint8_t *)dev_ptr);    \
        EXPECT_EQ(aclrtSynchronizeStream(stream), 0);                                                              \
                                                                                                                   \
        EXPECT_EQ(aclrtMemcpy(y_host, 1 * sizeof(TYPE), dev_ptr, 1 * sizeof(TYPE), ACL_MEMCPY_DEVICE_TO_HOST), 0); \
                                                                                                                   \
        /* result check */                                                                                         \
        flag = 0;                                                                                                  \
        if (y_host[0] != static_cast<TYPE>(test_offset + rank_id % rank_size))                                     \
            flag = 1;                                                                                              \
                                                                                                                   \
        EXPECT_EQ(aclrtFreeHost(y_host), 0);                                                                       \
        return flag;                                                                                               \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_SCALAR_PUT_GET);

#define TEST_ACLSHMEM_SCALAR(NAME, TYPE)                                                    \
    void test_##NAME##_aclshmem_scalar(int rank_id, int n_ranks, uint64_t local_mem_size)   \
    {                                                                                       \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                       \
        aclrtStream stream;                                                                 \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                               \
        ASSERT_NE(stream, nullptr);                                                         \
                                                                                            \
        int status = test_##NAME##_scalar_put_get(stream, rank_id, n_ranks);                \
        ASSERT_EQ(status, 0);                                                               \
                                                                                            \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;        \
        test_finalize(stream, device_id);                                                   \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_ACLSHMEM_SCALAR);

#define TEST_API(NAME, TYPE)                                                            \
    TEST(TestScalarApi, Test##NAME##ShmemScalar)                                        \
    {                                                                                   \
        const int process_count = test_gnpu_num;                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                               \
        test_mutil_task(test_##NAME##_aclshmem_scalar, local_mem_size, process_count);  \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_API);
