/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <iostream>
#include <string>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "p2p_kernel.h"

#define ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(FUNC)                                                                          \
    FUNC(float, float, 10.0f);                                                                                         \
    FUNC(int8, int8_t, 10);                                                                                            \
    FUNC(int16, int16_t, 10);                                                                                          \
    FUNC(int32, int32_t, 10);                                                                                          \
    FUNC(int64, int64_t, 10);                                                                                          \
    FUNC(uint8, uint8_t, 10);                                                                                          \
    FUNC(uint16, uint16_t, 10);                                                                                        \
    FUNC(uint32, uint32_t, 10);                                                                                        \
    FUNC(uint64, uint64_t, 10);                                                                                        \
    FUNC(char, char, 'b')

static void test_p2p(int rank_id, int rank_size, uint64_t local_mem_size)
{
    aclrtStream stream;
    test_init(rank_id, rank_size, local_mem_size, &stream);

    int32_t *addr_dev = static_cast<int32_t *>(aclshmem_malloc(sizeof(int32_t)));
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(int32_t), 0, sizeof(int32_t)), 0);
    p2p_chain_do(stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmem_free(addr_dev);

    int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;
    test_finalize(stream, dev_id);
}

TEST(TEST_SYNC_API, test_p2p)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_p2p, local_mem_size, process_count);
}

#define UNITTEST_P2P_WAIT_UNTIL(NAME, TYPE, CMP_VAL)                                                                   \
    static void test_p2p_##NAME##_wait_until(int rank_id, int rank_size, uint64_t local_mem_size)                      \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t input_size = sizeof(TYPE);                                                                              \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        *y_host = CMP_VAL;                                                                                             \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        p2p_chain_do_##NAME##_wait_until(stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size);     \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        aclshmem_free(addr_dev);                                                                                       \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_wait_until)                                                                  \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_wait_until, local_mem_size, process_count);                                  \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_WAIT_UNTIL);

#define UNITTEST_P2P_WAIT_UNTIL_ALL(NAME, TYPE, CMP_VAL)                                                               \
    static void test_p2p_##NAME##_wait_until_all(int rank_id, int rank_size, uint64_t local_mem_size)                  \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t nelems = 5;                                                                                             \
        size_t input_size = sizeof(TYPE) * nelems;                                                                     \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        std::fill(y_host, y_host + nelems, CMP_VAL);                                                                   \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t status_size = sizeof(int) * nelems;                                                                     \
        int *status = static_cast<int *>(aclshmem_malloc(status_size));                                                \
        ASSERT_EQ(aclrtMemset(status, status_size, 0, status_size), 0);                                                \
        int *status_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&status_host), status_size), 0);                           \
        ASSERT_EQ(aclrtMemset(status_host, status_size, 0, status_size), 0);                                           \
        std::fill(status_host, status_host + nelems, 0);                                                               \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(status, status_size, status_host, status_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                   \
        );                                                                                                             \
                                                                                                                       \
        p2p_chain_do_##NAME##_wait_until_all(                                                                          \
            stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, nelems, status                    \
        );                                                                                                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
                                                                                                                       \
        aclshmem_free(addr_dev);                                                                                       \
        aclshmem_free(status);                                                                                         \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFreeHost(status_host), 0);                                                                      \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_wait_until_all)                                                              \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_wait_until_all, local_mem_size, process_count);                              \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_WAIT_UNTIL_ALL);

#define UNITTEST_P2P_WAIT_UNTIL_ANY(NAME, TYPE, CMP_VAL)                                                               \
    static void test_p2p_##NAME##_wait_until_any(int rank_id, int rank_size, uint64_t local_mem_size)                  \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t nelems = 5;                                                                                             \
        size_t input_size = sizeof(TYPE) * nelems;                                                                     \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        y_host[0] = CMP_VAL;                                                                                           \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t status_size = sizeof(int) * nelems;                                                                     \
        int *status = static_cast<int *>(aclshmem_malloc(status_size));                                                \
        ASSERT_EQ(aclrtMemset(status, status_size, 0, status_size), 0);                                                \
        int *status_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&status_host), status_size), 0);                           \
        ASSERT_EQ(aclrtMemset(status_host, status_size, 0, status_size), 0);                                           \
        std::fill(status_host, status_host + nelems, 0);                                                               \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(status, status_size, status_host, status_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                   \
        );                                                                                                             \
                                                                                                                       \
        p2p_chain_do_##NAME##_wait_until_any(                                                                          \
            stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, nelems, status                    \
        );                                                                                                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
                                                                                                                       \
        aclshmem_free(addr_dev);                                                                                       \
        aclshmem_free(status);                                                                                         \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFreeHost(status_host), 0);                                                                      \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_wait_until_any)                                                              \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_wait_until_any, local_mem_size, process_count);                              \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_WAIT_UNTIL_ANY);

#define UNITTEST_P2P_WAIT_UNTIL_SOME(NAME, TYPE, CMP_VAL)                                                              \
    static void test_p2p_##NAME##_wait_until_some(int rank_id, int rank_size, uint64_t local_mem_size)                 \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t nelems = 5;                                                                                             \
        size_t input_size = sizeof(TYPE) * nelems;                                                                     \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        y_host[2] = CMP_VAL;                                                                                           \
        y_host[3] = CMP_VAL;                                                                                           \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t status_size = sizeof(int) * nelems;                                                                     \
        int *status = static_cast<int *>(aclshmem_malloc(status_size));                                                \
        ASSERT_EQ(aclrtMemset(status, status_size, 0, status_size), 0);                                                \
        int *status_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&status_host), status_size), 0);                           \
        ASSERT_EQ(aclrtMemset(status_host, status_size, 0, status_size), 0);                                           \
        std::fill(status_host, status_host + nelems, 0);                                                               \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(status, status_size, status_host, status_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                   \
        );                                                                                                             \
                                                                                                                       \
        size_t indices_size = sizeof(size_t) * nelems;                                                                 \
        size_t *indices = static_cast<size_t *>(aclshmem_malloc(indices_size));                                        \
        ASSERT_EQ(aclrtMemset(indices, indices_size, 0, indices_size), 0);                                             \
        size_t *indices_host = nullptr;                                                                                \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&indices_host), indices_size), 0);                         \
        ASSERT_EQ(aclrtMemset(indices_host, indices_size, 0, indices_size), 0);                                        \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(indices, indices_size, indices_host, indices_size, ACL_MEMCPY_HOST_TO_DEVICE), 0               \
        );                                                                                                             \
                                                                                                                       \
        p2p_chain_do_##NAME##_wait_until_some(                                                                         \
            stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, nelems, status, indices           \
        );                                                                                                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
                                                                                                                       \
        aclshmem_free(addr_dev);                                                                                       \
        aclshmem_free(status);                                                                                         \
        aclshmem_free(indices);                                                                                        \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFreeHost(status_host), 0);                                                                      \
        ASSERT_EQ(aclrtFreeHost(indices_host), 0);                                                                     \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_wait_until_some)                                                             \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_wait_until_some, local_mem_size, process_count);                             \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_WAIT_UNTIL_SOME);

#define UNITTEST_P2P_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE, CMP_VAL)                                                        \
    static void test_p2p_##NAME##_wait_until_all_vector(int rank_id, int rank_size, uint64_t local_mem_size)           \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t nelems = 5;                                                                                             \
        size_t input_size = sizeof(TYPE) * nelems;                                                                     \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        std::fill(y_host, y_host + nelems, CMP_VAL);                                                                   \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t status_size = sizeof(int) * nelems;                                                                     \
        int *status = static_cast<int *>(aclshmem_malloc(status_size));                                                \
        ASSERT_EQ(aclrtMemset(status, status_size, 0, status_size), 0);                                                \
        int *status_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&status_host), status_size), 0);                           \
        ASSERT_EQ(aclrtMemset(status_host, status_size, 0, status_size), 0);                                           \
        std::fill(status_host, status_host + nelems, 0);                                                               \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(status, status_size, status_host, status_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                   \
        );                                                                                                             \
                                                                                                                       \
        size_t cmp_values_size = sizeof(TYPE) * nelems * 3;                                                            \
        TYPE *cmp_values_addr = static_cast<TYPE *>(aclshmem_malloc(cmp_values_size));                                 \
        ASSERT_EQ(aclrtMemset(cmp_values_addr, cmp_values_size, 0, cmp_values_size), 0);                               \
        TYPE *cmp_values_host = nullptr;                                                                               \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&cmp_values_host), cmp_values_size), 0);                   \
        ASSERT_EQ(aclrtMemset(cmp_values_host, cmp_values_size, 0, cmp_values_size), 0);                               \
        std::fill(cmp_values_host, cmp_values_host + nelems, CMP_VAL);                                                 \
        std::fill(cmp_values_host + nelems, cmp_values_host + nelems * 2, (CMP_VAL) - 1);                              \
        std::fill(cmp_values_host + nelems * 2, cmp_values_host + nelems * 3, (CMP_VAL) + 1);                          \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(                                                                                               \
                cmp_values_addr, cmp_values_size, cmp_values_host, cmp_values_size, ACL_MEMCPY_HOST_TO_DEVICE          \
            ),                                                                                                         \
            0                                                                                                          \
        );                                                                                                             \
                                                                                                                       \
        p2p_chain_do_##NAME##_wait_until_all_vector(                                                                   \
            stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, nelems, status, cmp_values_addr   \
        );                                                                                                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        aclshmem_free(addr_dev);                                                                                       \
        aclshmem_free(status);                                                                                         \
        aclshmem_free(cmp_values_addr);                                                                                \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFreeHost(status_host), 0);                                                                      \
        ASSERT_EQ(aclrtFreeHost(cmp_values_host), 0);                                                                  \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_wait_until_all_vector)                                                       \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_wait_until_all_vector, local_mem_size, process_count);                       \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_WAIT_UNTIL_ALL_VECTOR);

#define UNITTEST_P2P_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE, CMP_VAL)                                                        \
    static void test_p2p_##NAME##_wait_until_any_vector(int rank_id, int rank_size, uint64_t local_mem_size)           \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t nelems = 5;                                                                                             \
        size_t input_size = sizeof(TYPE) * nelems;                                                                     \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        y_host[0] = CMP_VAL;                                                                                           \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t status_size = sizeof(int) * nelems;                                                                     \
        int *status = static_cast<int *>(aclshmem_malloc(status_size));                                                \
        ASSERT_EQ(aclrtMemset(status, status_size, 0, status_size), 0);                                                \
        int *status_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&status_host), status_size), 0);                           \
        ASSERT_EQ(aclrtMemset(status_host, status_size, 0, status_size), 0);                                           \
        std::fill(status_host, status_host + nelems, 0);                                                               \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(status, status_size, status_host, status_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                   \
        );                                                                                                             \
                                                                                                                       \
        size_t cmp_values_size = sizeof(TYPE) * nelems * 3;                                                            \
        TYPE *cmp_values_addr = static_cast<TYPE *>(aclshmem_malloc(cmp_values_size));                                 \
        ASSERT_EQ(aclrtMemset(cmp_values_addr, cmp_values_size, 0, cmp_values_size), 0);                               \
        TYPE *cmp_values_host = nullptr;                                                                               \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&cmp_values_host), cmp_values_size), 0);                   \
        ASSERT_EQ(aclrtMemset(cmp_values_host, cmp_values_size, 0, cmp_values_size), 0);                               \
        std::fill(cmp_values_host, cmp_values_host + nelems, CMP_VAL);                                                 \
        std::fill(cmp_values_host + nelems, cmp_values_host + nelems * 2, (CMP_VAL) - 1);                              \
        std::fill(cmp_values_host + nelems * 2, cmp_values_host + nelems * 3, (CMP_VAL) + 1);                          \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(                                                                                               \
                cmp_values_addr, cmp_values_size, cmp_values_host, cmp_values_size, ACL_MEMCPY_HOST_TO_DEVICE          \
            ),                                                                                                         \
            0                                                                                                          \
        );                                                                                                             \
                                                                                                                       \
        p2p_chain_do_##NAME##_wait_until_any_vector(                                                                   \
            stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, nelems, status, cmp_values_addr   \
        );                                                                                                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        aclshmem_free(addr_dev);                                                                                       \
        aclshmem_free(status);                                                                                         \
        aclshmem_free(cmp_values_addr);                                                                                \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFreeHost(status_host), 0);                                                                      \
        ASSERT_EQ(aclrtFreeHost(cmp_values_host), 0);                                                                  \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_wait_until_any_vector)                                                       \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_wait_until_any_vector, local_mem_size, process_count);                       \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_WAIT_UNTIL_ANY_VECTOR);

#define UNITTEST_P2P_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE, CMP_VAL)                                                       \
    static void test_p2p_##NAME##_wait_until_some_vector(int rank_id, int rank_size, uint64_t local_mem_size)          \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t nelems = 5;                                                                                             \
        size_t input_size = sizeof(TYPE) * nelems;                                                                     \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        y_host[2] = CMP_VAL;                                                                                           \
        y_host[3] = CMP_VAL;                                                                                           \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t status_size = sizeof(int) * nelems;                                                                     \
        int *status = static_cast<int *>(aclshmem_malloc(status_size));                                                \
        ASSERT_EQ(aclrtMemset(status, status_size, 0, status_size), 0);                                                \
        int *status_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&status_host), status_size), 0);                           \
        ASSERT_EQ(aclrtMemset(status_host, status_size, 0, status_size), 0);                                           \
        std::fill(status_host, status_host + nelems, 0);                                                               \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(status, status_size, status_host, status_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                   \
        );                                                                                                             \
                                                                                                                       \
        size_t indices_size = sizeof(size_t) * nelems;                                                                 \
        size_t *indices = static_cast<size_t *>(aclshmem_malloc(indices_size));                                        \
        ASSERT_EQ(aclrtMemset(indices, indices_size, 0, indices_size), 0);                                             \
        size_t *indices_host = nullptr;                                                                                \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&indices_host), indices_size), 0);                         \
        ASSERT_EQ(aclrtMemset(indices_host, indices_size, 0, indices_size), 0);                                        \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(indices, indices_size, indices_host, indices_size, ACL_MEMCPY_HOST_TO_DEVICE), 0               \
        );                                                                                                             \
                                                                                                                       \
        size_t cmp_values_size = sizeof(TYPE) * nelems * 3;                                                            \
        TYPE *cmp_values_addr = static_cast<TYPE *>(aclshmem_malloc(cmp_values_size));                                 \
        ASSERT_EQ(aclrtMemset(cmp_values_addr, cmp_values_size, 0, cmp_values_size), 0);                               \
        TYPE *cmp_values_host = nullptr;                                                                               \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&cmp_values_host), cmp_values_size), 0);                   \
        ASSERT_EQ(aclrtMemset(cmp_values_host, cmp_values_size, 0, cmp_values_size), 0);                               \
        std::fill(cmp_values_host, cmp_values_host + nelems, CMP_VAL);                                                 \
        std::fill(cmp_values_host + nelems, cmp_values_host + nelems * 2, (CMP_VAL) - 1);                              \
        std::fill(cmp_values_host + nelems * 2, cmp_values_host + nelems * 3, (CMP_VAL) + 1);                          \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(cmp_values_addr, cmp_values_size, cmp_values_host, cmp_values_size, ACL_MEMCPY_HOST_TO_DEVICE),\
            0                                                                                                          \
        );                                                                                                             \
                                                                                                                       \
        p2p_chain_do_##NAME##_wait_until_some_vector(                                                                  \
            stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, nelems, status, indices,          \
            cmp_values_addr                                                                                            \
        );                                                                                                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        aclshmem_free(addr_dev);                                                                                       \
        aclshmem_free(status);                                                                                         \
        aclshmem_free(indices);                                                                                        \
        aclshmem_free(cmp_values_addr);                                                                                \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFreeHost(status_host), 0);                                                                      \
        ASSERT_EQ(aclrtFreeHost(indices_host), 0);                                                                     \
        ASSERT_EQ(aclrtFreeHost(cmp_values_host), 0);                                                                  \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_wait_until_some_vector)                                                      \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_wait_until_some_vector, local_mem_size, process_count);                      \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_WAIT_UNTIL_SOME_VECTOR);

#define UNITTEST_P2P_TEST(NAME, TYPE, CMP_VAL)                                                                         \
    static void test_p2p_##NAME##_test(int rank_id, int rank_size, uint64_t local_mem_size)                            \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t input_size = sizeof(TYPE);                                                                              \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        *y_host = CMP_VAL;                                                                                             \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t res_size = sizeof(int) * 8;                                                                             \
        int *res_addr = nullptr;                                                                                       \
        EXPECT_EQ(aclrtMalloc(reinterpret_cast<void **>(&res_addr), res_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);          \
        ASSERT_EQ(aclrtMemset(res_addr, res_size, 0, res_size), 0);                                                    \
        int *res_host = nullptr;                                                                                       \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&res_host), res_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(res_host, res_size, 0, res_size), 0);                                                    \
                                                                                                                       \
        p2p_chain_do_##NAME##_test(stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, res_addr); \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(res_host, res_size, res_addr, res_size, ACL_MEMCPY_DEVICE_TO_HOST), 0                          \
        );                                                                                                             \
        for (size_t i = 0; i < 8; ++i) {                                                                               \
            ASSERT_EQ(res_host[i], 1);                                                                                 \
        }                                                                                                              \
        aclshmem_free(addr_dev);                                                                                       \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFree(res_addr), 0);                                                                             \
        ASSERT_EQ(aclrtFreeHost(res_host), 0);                                                                         \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_test)                                                                        \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_test, local_mem_size, process_count);                                        \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_TEST);

#define UNITTEST_P2P_TEST_ANY(NAME, TYPE, CMP_VAL)                                                                     \
    static void test_p2p_##NAME##_test_any(int rank_id, int rank_size, uint64_t local_mem_size)                        \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t nelems = 5;                                                                                             \
        size_t input_size = sizeof(TYPE) * nelems;                                                                     \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        y_host[2] = CMP_VAL;                                                                                           \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t status_size = sizeof(int) * nelems;                                                                     \
        int *status = static_cast<int *>(aclshmem_malloc(status_size));                                                \
        ASSERT_EQ(aclrtMemset(status, status_size, 0, status_size), 0);                                                \
        int *status_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&status_host), status_size), 0);                           \
        ASSERT_EQ(aclrtMemset(status_host, status_size, 0, status_size), 0);                                           \
        std::fill(status_host, status_host + nelems, 0);                                                               \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(status, status_size, status_host, status_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                   \
        );                                                                                                             \
                                                                                                                       \
        size_t res_size = sizeof(size_t) * 8;                                                                          \
        size_t *res_addr = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMalloc(reinterpret_cast<void **>(&res_addr), res_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);          \
        ASSERT_EQ(aclrtMemset(res_addr, res_size, 0, res_size), 0);                                                    \
        size_t *res_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&res_host), res_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(res_host, res_size, 0, res_size), 0);                                                    \
                                                                                                                       \
        p2p_chain_do_##NAME##_test_any(                                                                                \
            stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, nelems, status, res_addr          \
        );                                                                                                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(res_host, res_size, res_addr, res_size, ACL_MEMCPY_DEVICE_TO_HOST), 0                          \
        );                                                                                                             \
        ASSERT_EQ(res_host[0], 2);                                                                                     \
        ASSERT_EQ(res_host[1], 0);                                                                                     \
        for (size_t i = 2; i < 5; ++i) {                                                                               \
            ASSERT_EQ(res_host[i], 2);                                                                                 \
        }                                                                                                              \
        for (size_t i = 5; i < 8; ++i) {                                                                               \
            ASSERT_EQ(res_host[i], 0);                                                                                 \
        }                                                                                                              \
                                                                                                                       \
        aclshmem_free(addr_dev);                                                                                       \
        aclshmem_free(status);                                                                                         \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFreeHost(status_host), 0);                                                                      \
        ASSERT_EQ(aclrtFree(res_addr), 0);                                                                             \
        ASSERT_EQ(aclrtFreeHost(res_host), 0);                                                                         \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_test_any)                                                                    \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_test_any, local_mem_size, process_count);                                    \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_TEST_ANY);

#define UNITTEST_P2P_TEST_SOME(NAME, TYPE, CMP_VAL)                                                                    \
    static void test_p2p_##NAME##_test_some(int rank_id, int rank_size, uint64_t local_mem_size)                       \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t nelems = 5;                                                                                             \
        size_t input_size = sizeof(TYPE) * nelems;                                                                     \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        y_host[2] = CMP_VAL;                                                                                           \
        y_host[3] = CMP_VAL;                                                                                           \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t status_size = sizeof(int) * nelems;                                                                     \
        int *status = static_cast<int *>(aclshmem_malloc(status_size));                                                \
        ASSERT_EQ(aclrtMemset(status, status_size, 0, status_size), 0);                                                \
        int *status_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&status_host), status_size), 0);                           \
        ASSERT_EQ(aclrtMemset(status_host, status_size, 0, status_size), 0);                                           \
        std::fill(status_host, status_host + nelems, 0);                                                               \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(status, status_size, status_host, status_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                   \
        );                                                                                                             \
                                                                                                                       \
        size_t indices_size = sizeof(size_t) * nelems;                                                                 \
        size_t *indices = static_cast<size_t *>(aclshmem_malloc(indices_size));                                        \
        ASSERT_EQ(aclrtMemset(indices, indices_size, 0, indices_size), 0);                                             \
        size_t *indices_host = nullptr;                                                                                \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&indices_host), indices_size), 0);                         \
        ASSERT_EQ(aclrtMemset(indices_host, indices_size, 0, indices_size), 0);                                        \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(indices, indices_size, indices_host, indices_size, ACL_MEMCPY_HOST_TO_DEVICE), 0               \
        );                                                                                                             \
                                                                                                                       \
        size_t res_size = sizeof(size_t) * 8;                                                                          \
        size_t *res_addr = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMalloc(reinterpret_cast<void **>(&res_addr), res_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);          \
        ASSERT_EQ(aclrtMemset(res_addr, res_size, 0, res_size), 0);                                                    \
        size_t *res_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&res_host), res_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(res_host, res_size, 0, res_size), 0);                                                    \
                                                                                                                       \
        p2p_chain_do_##NAME##_test_some(                                                                               \
            stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, nelems, status, indices, res_addr \
        );                                                                                                             \
                                                                                                                       \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(res_host, res_size, res_addr, res_size, ACL_MEMCPY_DEVICE_TO_HOST), 0                          \
        );                                                                                                             \
        ASSERT_EQ(res_host[0], 2);                                                                                     \
        ASSERT_EQ(res_host[1], 3);                                                                                     \
        for (size_t i = 2; i < 5; ++i) {                                                                               \
            ASSERT_EQ(res_host[i], 2);                                                                                 \
        }                                                                                                              \
        for (size_t i = 5; i < 8; ++i) {                                                                               \
            ASSERT_EQ(res_host[i], 5);                                                                                 \
        }                                                                                                              \
                                                                                                                       \
        aclshmem_free(addr_dev);                                                                                       \
        aclshmem_free(status);                                                                                         \
        aclshmem_free(indices);                                                                                        \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFreeHost(status_host), 0);                                                                      \
        ASSERT_EQ(aclrtFreeHost(indices_host), 0);                                                                     \
        ASSERT_EQ(aclrtFree(res_addr), 0);                                                                             \
        ASSERT_EQ(aclrtFreeHost(res_host), 0);                                                                         \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_test_some)                                                                   \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_test_some, local_mem_size, process_count);                                   \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_TEST_SOME);

#define UNITTEST_P2P_TEST_ALL_VECTOR(NAME, TYPE, CMP_VAL)                                                              \
    static void test_p2p_##NAME##_test_all_vector(int rank_id, int rank_size, uint64_t local_mem_size)                 \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t nelems = 5;                                                                                             \
        size_t input_size = sizeof(TYPE) * nelems;                                                                     \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        std::fill(y_host, y_host + nelems, CMP_VAL);                                                                   \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t status_size = sizeof(int) * nelems;                                                                     \
        int *status = static_cast<int *>(aclshmem_malloc(status_size));                                                \
        ASSERT_EQ(aclrtMemset(status, status_size, 0, status_size), 0);                                                \
        int *status_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&status_host), status_size), 0);                           \
        ASSERT_EQ(aclrtMemset(status_host, status_size, 0, status_size), 0);                                           \
        std::fill(status_host, status_host + nelems, 0);                                                               \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(status, status_size, status_host, status_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                   \
        );                                                                                                             \
                                                                                                                       \
        size_t cmp_values_size = sizeof(TYPE) * nelems * 3;                                                            \
        TYPE *cmp_values_addr = static_cast<TYPE *>(aclshmem_malloc(cmp_values_size));                                 \
        ASSERT_EQ(aclrtMemset(cmp_values_addr, cmp_values_size, 0, cmp_values_size), 0);                               \
        TYPE *cmp_values_host = nullptr;                                                                               \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&cmp_values_host), cmp_values_size), 0);                   \
        ASSERT_EQ(aclrtMemset(cmp_values_host, cmp_values_size, 0, cmp_values_size), 0);                               \
        std::fill(cmp_values_host, cmp_values_host + nelems, CMP_VAL);                                                 \
        std::fill(cmp_values_host + nelems, cmp_values_host + nelems * 2, (CMP_VAL) - 1);                              \
        std::fill(cmp_values_host + nelems * 2, cmp_values_host + nelems * 3, (CMP_VAL) + 1);                          \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(                                                                                               \
                cmp_values_addr, cmp_values_size, cmp_values_host, cmp_values_size, ACL_MEMCPY_HOST_TO_DEVICE          \
            ),                                                                                                         \
            0                                                                                                          \
        );                                                                                                             \
                                                                                                                       \
        size_t res_size = sizeof(int) * 8;                                                                             \
        int *res_addr = nullptr;                                                                                       \
        EXPECT_EQ(aclrtMalloc(reinterpret_cast<void **>(&res_addr), res_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);          \
        ASSERT_EQ(aclrtMemset(res_addr, res_size, 0, res_size), 0);                                                    \
        int *res_host = nullptr;                                                                                       \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&res_host), res_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(res_host, res_size, 0, res_size), 0);                                                    \
                                                                                                                       \
        p2p_chain_do_##NAME##_test_all_vector(                                                                         \
            stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, nelems, status, cmp_values_addr,  \
            res_addr                                                                                                   \
        );                                                                                                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(res_host, res_size, res_addr, res_size, ACL_MEMCPY_DEVICE_TO_HOST), 0                          \
        );                                                                                                             \
        for (size_t i = 0; i < 8; ++i) {                                                                               \
            ASSERT_EQ(res_host[i], 1);                                                                                 \
        }                                                                                                              \
                                                                                                                       \
        aclshmem_free(addr_dev);                                                                                       \
        aclshmem_free(status);                                                                                         \
        aclshmem_free(cmp_values_addr);                                                                                \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFreeHost(status_host), 0);                                                                      \
        ASSERT_EQ(aclrtFreeHost(cmp_values_host), 0);                                                                  \
        ASSERT_EQ(aclrtFree(res_addr), 0);                                                                             \
        ASSERT_EQ(aclrtFreeHost(res_host), 0);                                                                         \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_test_all_vector)                                                             \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_test_all_vector, local_mem_size, process_count);                             \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_TEST_ALL_VECTOR);

#define UNITTEST_P2P_TEST_ANY_VECTOR(NAME, TYPE, CMP_VAL)                                                              \
    static void test_p2p_##NAME##_test_any_vector(int rank_id, int rank_size, uint64_t local_mem_size)                 \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t nelems = 5;                                                                                             \
        size_t input_size = sizeof(TYPE) * nelems;                                                                     \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        y_host[2] = CMP_VAL;                                                                                           \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t status_size = sizeof(int) * nelems;                                                                     \
        int *status = static_cast<int *>(aclshmem_malloc(status_size));                                                \
        ASSERT_EQ(aclrtMemset(status, status_size, 0, status_size), 0);                                                \
        int *status_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&status_host), status_size), 0);                           \
        ASSERT_EQ(aclrtMemset(status_host, status_size, 0, status_size), 0);                                           \
        std::fill(status_host, status_host + nelems, 0);                                                               \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(status, status_size, status_host, status_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                   \
        );                                                                                                             \
                                                                                                                       \
        size_t cmp_values_size = sizeof(TYPE) * nelems * 3;                                                            \
        TYPE *cmp_values_addr = static_cast<TYPE *>(aclshmem_malloc(cmp_values_size));                                 \
        ASSERT_EQ(aclrtMemset(cmp_values_addr, cmp_values_size, 0, cmp_values_size), 0);                               \
        TYPE *cmp_values_host = nullptr;                                                                               \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&cmp_values_host), cmp_values_size), 0);                   \
        ASSERT_EQ(aclrtMemset(cmp_values_host, cmp_values_size, 0, cmp_values_size), 0);                               \
        std::fill(cmp_values_host, cmp_values_host + nelems, CMP_VAL);                                                 \
        std::fill(cmp_values_host + nelems, cmp_values_host + nelems * 2, (CMP_VAL) - 1);                              \
        std::fill(cmp_values_host + nelems * 2, cmp_values_host + nelems * 3, (CMP_VAL) + 1);                          \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(                                                                                               \
                cmp_values_addr, cmp_values_size, cmp_values_host, cmp_values_size, ACL_MEMCPY_HOST_TO_DEVICE          \
            ),                                                                                                         \
            0                                                                                                          \
        );                                                                                                             \
                                                                                                                       \
        size_t res_size = sizeof(size_t) * 8;                                                                          \
        size_t *res_addr = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMalloc(reinterpret_cast<void **>(&res_addr), res_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);          \
        ASSERT_EQ(aclrtMemset(res_addr, res_size, 0, res_size), 0);                                                    \
        size_t *res_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&res_host), res_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(res_host, res_size, 0, res_size), 0);                                                    \
                                                                                                                       \
        p2p_chain_do_##NAME##_test_any_vector(                                                                         \
            stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, nelems, status, cmp_values_addr,  \
            res_addr                                                                                                   \
        );                                                                                                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(res_host, res_size, res_addr, res_size, ACL_MEMCPY_DEVICE_TO_HOST), 0                          \
        );                                                                                                             \
        ASSERT_EQ(res_host[0], 2);                                                                                     \
        ASSERT_EQ(res_host[1], 0);                                                                                     \
        for (size_t i = 2; i < 5; ++i) {                                                                               \
            ASSERT_EQ(res_host[i], 2);                                                                                 \
        }                                                                                                              \
        for (size_t i = 5; i < 8; ++i) {                                                                               \
            ASSERT_EQ(res_host[i], 0);                                                                                 \
        }                                                                                                              \
                                                                                                                       \
        aclshmem_free(addr_dev);                                                                                       \
        aclshmem_free(status);                                                                                         \
        aclshmem_free(cmp_values_addr);                                                                                \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFreeHost(status_host), 0);                                                                      \
        ASSERT_EQ(aclrtFreeHost(cmp_values_host), 0);                                                                  \
        ASSERT_EQ(aclrtFree(res_addr), 0);                                                                             \
        ASSERT_EQ(aclrtFreeHost(res_host), 0);                                                                         \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_test_any_vector)                                                             \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_test_any_vector, local_mem_size, process_count);                             \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_TEST_ANY_VECTOR);

#define UNITTEST_P2P_TEST_SOME_VECTOR(NAME, TYPE, CMP_VAL)                                                             \
    static void test_p2p_##NAME##_test_some_vector(int rank_id, int rank_size, uint64_t local_mem_size)                \
    {                                                                                                                  \
        aclrtStream stream;                                                                                            \
        test_init(rank_id, rank_size, local_mem_size, &stream);                                                        \
                                                                                                                       \
        size_t nelems = 5;                                                                                             \
        size_t input_size = sizeof(TYPE) * nelems;                                                                     \
        TYPE *addr_dev = static_cast<TYPE *>(aclshmem_malloc(input_size));                                             \
        ASSERT_EQ(aclrtMemset(addr_dev, input_size, 0, input_size), 0);                                                \
        TYPE *y_host = nullptr;                                                                                        \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(y_host, input_size, 0, input_size), 0);                                                  \
        y_host[2] = CMP_VAL;                                                                                           \
        y_host[3] = CMP_VAL;                                                                                           \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(addr_dev, input_size, y_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                        \
        );                                                                                                             \
                                                                                                                       \
        size_t status_size = sizeof(int) * nelems;                                                                     \
        int *status = static_cast<int *>(aclshmem_malloc(status_size));                                                \
        ASSERT_EQ(aclrtMemset(status, status_size, 0, status_size), 0);                                                \
        int *status_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&status_host), status_size), 0);                           \
        ASSERT_EQ(aclrtMemset(status_host, status_size, 0, status_size), 0);                                           \
        std::fill(status_host, status_host + nelems, 0);                                                               \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(status, status_size, status_host, status_size, ACL_MEMCPY_HOST_TO_DEVICE), 0                   \
        );                                                                                                             \
                                                                                                                       \
        size_t indices_size = sizeof(size_t) * nelems;                                                                 \
        size_t *indices = static_cast<size_t *>(aclshmem_malloc(indices_size));                                        \
        ASSERT_EQ(aclrtMemset(indices, indices_size, 0, indices_size), 0);                                             \
        size_t *indices_host = nullptr;                                                                                \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&indices_host), indices_size), 0);                         \
        ASSERT_EQ(aclrtMemset(indices_host, indices_size, 0, indices_size), 0);                                        \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(indices, indices_size, indices_host, indices_size, ACL_MEMCPY_HOST_TO_DEVICE), 0               \
        );                                                                                                             \
                                                                                                                       \
        size_t cmp_values_size = sizeof(TYPE) * nelems * 3;                                                            \
        TYPE *cmp_values_addr = static_cast<TYPE *>(aclshmem_malloc(cmp_values_size));                                 \
        ASSERT_EQ(aclrtMemset(cmp_values_addr, cmp_values_size, 0, cmp_values_size), 0);                               \
        TYPE *cmp_values_host = nullptr;                                                                               \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&cmp_values_host), cmp_values_size), 0);                   \
        ASSERT_EQ(aclrtMemset(cmp_values_host, cmp_values_size, 0, cmp_values_size), 0);                               \
        std::fill(cmp_values_host, cmp_values_host + nelems, CMP_VAL);                                                 \
        std::fill(cmp_values_host + nelems, cmp_values_host + nelems * 2, (CMP_VAL) - 1);                              \
        std::fill(cmp_values_host + nelems * 2, cmp_values_host + nelems * 3, (CMP_VAL) + 1);                          \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(                                                                                               \
                cmp_values_addr, cmp_values_size, cmp_values_host, cmp_values_size, ACL_MEMCPY_HOST_TO_DEVICE          \
            ),                                                                                                         \
            0                                                                                                          \
        );                                                                                                             \
                                                                                                                       \
        size_t res_size = sizeof(size_t) * 8;                                                                          \
        size_t *res_addr = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMalloc(reinterpret_cast<void **>(&res_addr), res_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);          \
        ASSERT_EQ(aclrtMemset(res_addr, res_size, 0, res_size), 0);                                                    \
        size_t *res_host = nullptr;                                                                                    \
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&res_host), res_size), 0);                                 \
        ASSERT_EQ(aclrtMemset(res_host, res_size, 0, res_size), 0);                                                    \
                                                                                                                       \
        p2p_chain_do_##NAME##_test_some_vector(                                                                        \
            stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, rank_size, nelems, status, indices,          \
            cmp_values_addr, res_addr                                                                                  \
        );                                                                                                             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        ASSERT_EQ(                                                                                                     \
            aclrtMemcpy(res_host, res_size, res_addr, res_size, ACL_MEMCPY_DEVICE_TO_HOST), 0                          \
        );                                                                                                             \
        ASSERT_EQ(res_host[0], 2);                                                                                     \
        ASSERT_EQ(res_host[1], 3);                                                                                     \
        for (size_t i = 2; i < 5; ++i) {                                                                               \
            ASSERT_EQ(res_host[i], 2);                                                                                 \
        }                                                                                                              \
        for (size_t i = 5; i < 8; ++i) {                                                                               \
            ASSERT_EQ(res_host[i], 5);                                                                                 \
        }                                                                                                              \
                                                                                                                       \
        aclshmem_free(addr_dev);                                                                                       \
        aclshmem_free(status);                                                                                         \
        aclshmem_free(indices);                                                                                        \
        aclshmem_free(cmp_values_addr);                                                                                \
        ASSERT_EQ(aclrtFreeHost(y_host), 0);                                                                           \
        ASSERT_EQ(aclrtFreeHost(status_host), 0);                                                                      \
        ASSERT_EQ(aclrtFreeHost(indices_host), 0);                                                                     \
        ASSERT_EQ(aclrtFreeHost(cmp_values_host), 0);                                                                  \
        ASSERT_EQ(aclrtFree(res_addr), 0);                                                                             \
        ASSERT_EQ(aclrtFreeHost(res_host), 0);                                                                         \
                                                                                                                       \
        int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;                                                     \
        test_finalize(stream, dev_id);                                                                                 \
    }                                                                                                                  \
                                                                                                                       \
    TEST(TEST_SYNC_API, test_p2p_##NAME##_test_some_vector)                                                            \
    {                                                                                                                  \
        const int32_t process_count = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 16;                                                                \
        test_mutil_task(test_p2p_##NAME##_test_some_vector, local_mem_size, process_count);                            \
    }

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(UNITTEST_P2P_TEST_SOME_VECTOR);