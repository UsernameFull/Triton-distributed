/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <cstdint>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "func_type.h"
#include "unittest_main_test.h"

// ============================================================
// Atomic Add
// ============================================================
#define TEST_UDMA_ATOMIC_ADD_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_add_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_ADD_FUNC_TYPE(TEST_UDMA_ATOMIC_ADD_FUNC);

#define TEST_UDMA_ATOMIC_ADD_HOST(NAME, TYPE) \
    static void test_udma_atomic_add_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,                 \
                                                   uint32_t rank_size)                                                 \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = sizeof(TYPE);                                                                               \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = rank_id;                                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_add_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                                    \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(rank_id + 10));                                                          \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_ADD_FUNC_TYPE(TEST_UDMA_ATOMIC_ADD_HOST);

#define TEST_UDMA_ATOMIC_ADD_MEM(NAME, TYPE)                                                                           \
    void test_udma_atomic_add_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                          \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_add_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);                    \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_ADD_FUNC_TYPE(TEST_UDMA_ATOMIC_ADD_MEM);


// ============================================================
// Atomic Fetch Add
// ============================================================
#define TEST_UDMA_ATOMIC_FETCH_ADD_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_fetch_add_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_ADD_FUNC);

#define TEST_UDMA_ATOMIC_FETCH_ADD_HOST(NAME, TYPE)                                                                    \
    static void test_udma_atomic_fetch_add_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,           \
                                                         uint32_t rank_size)                                           \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = 2 * sizeof(TYPE);                                                                           \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = rank_id;                                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_fetch_add_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                              \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(rank_id + 10));                                                          \
        ASSERT_EQ(xHost[1], static_cast<TYPE>((rank_id + 1) % rank_size));                                             \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_ADD_HOST);

#define TEST_UDMA_ATOMIC_FETCH_ADD_MEM(NAME, TYPE)                                                                     \
    void test_udma_atomic_fetch_add_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                    \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_fetch_add_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);              \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_ADD_MEM);


// ============================================================
// Atomic Compare Swap
// ============================================================
#define TEST_UDMA_ATOMIC_COMPARE_SWAP_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_compare_swap_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_COMPARE_SWAP_FUNC);

#define TEST_UDMA_ATOMIC_COMPARE_SWAP_HOST(NAME, TYPE)                                                                 \
    static void test_udma_atomic_compare_swap_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,        \
                                                             uint32_t rank_size)                                       \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = 2 * sizeof(TYPE);                                                                           \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = 1;                                                                                                  \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_compare_swap_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                           \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(10));                                                                    \
        ASSERT_EQ(xHost[1], static_cast<TYPE>(1));                                                                     \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_COMPARE_SWAP_HOST);

#define TEST_UDMA_ATOMIC_COMPARE_SWAP_MEM(NAME, TYPE)                                                                  \
    void test_udma_atomic_compare_swap_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                 \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_compare_swap_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);           \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_COMPARE_SWAP_MEM);


// ============================================================
// Atomic Fetch
// ============================================================
#define TEST_UDMA_ATOMIC_FETCH_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_fetch_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_FUNC);

#define TEST_UDMA_ATOMIC_FETCH_HOST(NAME, TYPE)                                                                        \
    static void test_udma_atomic_fetch_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,               \
                                                     uint32_t rank_size)                                               \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = 2 * sizeof(TYPE);                                                                           \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = rank_id;                                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_fetch_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                                  \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(rank_id));                                                               \
        ASSERT_EQ(xHost[1], static_cast<TYPE>((rank_id + 1) % rank_size));                                             \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_HOST);

#define TEST_UDMA_ATOMIC_FETCH_MEM(NAME, TYPE)                                                                         \
    void test_udma_atomic_fetch_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                        \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_fetch_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);                  \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_MEM);


// ============================================================
// Atomic Set
// ============================================================
#define TEST_UDMA_ATOMIC_SET_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_set_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_SET_FUNC);

#define TEST_UDMA_ATOMIC_SET_HOST(NAME, TYPE)                                                                          \
    static void test_udma_atomic_set_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,                 \
                                                   uint32_t rank_size)                                                 \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = sizeof(TYPE);                                                                               \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = rank_id;                                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_set_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                                    \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(100));                                                                   \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_SET_HOST);

#define TEST_UDMA_ATOMIC_SET_MEM(NAME, TYPE)                                                                           \
    void test_udma_atomic_set_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                          \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_set_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);                    \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_SET_MEM);


// ============================================================
// Atomic Swap
// ============================================================
#define TEST_UDMA_ATOMIC_SWAP_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_swap_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_SWAP_FUNC);

#define TEST_UDMA_ATOMIC_SWAP_HOST(NAME, TYPE)                                                                         \
    static void test_udma_atomic_swap_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,                \
                                                    uint32_t rank_size)                                                \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = 2 * sizeof(TYPE);                                                                           \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = rank_id;                                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_swap_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                                   \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(100));                                                                   \
        ASSERT_EQ(xHost[1], static_cast<TYPE>((rank_id + 1) % rank_size));                                             \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_SWAP_HOST);

#define TEST_UDMA_ATOMIC_SWAP_MEM(NAME, TYPE)                                                                          \
    void test_udma_atomic_swap_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                         \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_swap_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);                   \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_SWAP_MEM);


// ============================================================
// Atomic Fetch Inc
// ============================================================
#define TEST_UDMA_ATOMIC_FETCH_INC_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_fetch_inc_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_INC_FUNC);

#define TEST_UDMA_ATOMIC_FETCH_INC_HOST(NAME, TYPE)                                                                    \
    static void test_udma_atomic_fetch_inc_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,           \
                                                          uint32_t rank_size)                                          \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = 2 * sizeof(TYPE);                                                                           \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = rank_id;                                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_fetch_inc_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                              \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(rank_id + 1));                                                           \
        ASSERT_EQ(xHost[1], static_cast<TYPE>((rank_id + 1) % rank_size));                                             \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_INC_HOST);

#define TEST_UDMA_ATOMIC_FETCH_INC_MEM(NAME, TYPE)                                                                     \
    void test_udma_atomic_fetch_inc_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                    \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_fetch_inc_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);              \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_INC_MEM);


// ============================================================
// Atomic Inc
// ============================================================
#define TEST_UDMA_ATOMIC_INC_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_inc_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_INC_FUNC);

#define TEST_UDMA_ATOMIC_INC_HOST(NAME, TYPE)                                                                          \
    static void test_udma_atomic_inc_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,                 \
                                                    uint32_t rank_size)                                                \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = sizeof(TYPE);                                                                               \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = rank_id;                                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_inc_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                                    \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(rank_id + 1));                                                           \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_INC_HOST);

#define TEST_UDMA_ATOMIC_INC_MEM(NAME, TYPE)                                                                           \
    void test_udma_atomic_inc_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                          \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_inc_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);                    \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_INC_MEM);


// ============================================================
// Atomic Fetch And
// ============================================================
#define TEST_UDMA_ATOMIC_FETCH_AND_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_fetch_and_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_AND_FUNC);

#define TEST_UDMA_ATOMIC_FETCH_AND_HOST(NAME, TYPE)                                                                    \
    static void test_udma_atomic_fetch_and_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,           \
                                                          uint32_t rank_size)                                          \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = 2 * sizeof(TYPE);                                                                           \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = static_cast<TYPE>(0xFF);                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_fetch_and_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                              \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(0x0F));                                                                  \
        ASSERT_EQ(xHost[1], static_cast<TYPE>(0xFF));                                                                  \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_AND_HOST);

#define TEST_UDMA_ATOMIC_FETCH_AND_MEM(NAME, TYPE)                                                                     \
    void test_udma_atomic_fetch_and_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                    \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_fetch_and_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);              \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_AND_MEM);


// ============================================================
// Atomic And
// ============================================================
#define TEST_UDMA_ATOMIC_AND_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_and_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_AND_FUNC);

#define TEST_UDMA_ATOMIC_AND_HOST(NAME, TYPE)                                                                          \
    static void test_udma_atomic_and_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,                 \
                                                   uint32_t rank_size)                                                 \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = sizeof(TYPE);                                                                               \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = static_cast<TYPE>(0xFF);                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_and_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                                    \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(0x0F));                                                                  \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_AND_HOST);

#define TEST_UDMA_ATOMIC_AND_MEM(NAME, TYPE)                                                                           \
    void test_udma_atomic_and_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                          \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_and_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);                    \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_AND_MEM);


// ============================================================
// Atomic Fetch Or
// ============================================================
#define TEST_UDMA_ATOMIC_FETCH_OR_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_fetch_or_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_OR_FUNC);

#define TEST_UDMA_ATOMIC_FETCH_OR_HOST(NAME, TYPE)                                                                     \
    static void test_udma_atomic_fetch_or_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,            \
                                                         uint32_t rank_size)                                           \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = 2 * sizeof(TYPE);                                                                           \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = static_cast<TYPE>(0x0F);                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_fetch_or_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                               \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(0xFF));                                                                  \
        ASSERT_EQ(xHost[1], static_cast<TYPE>(0x0F));                                                                  \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_OR_HOST);

#define TEST_UDMA_ATOMIC_FETCH_OR_MEM(NAME, TYPE)                                                                      \
    void test_udma_atomic_fetch_or_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                     \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_fetch_or_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);               \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_OR_MEM);


// ============================================================
// Atomic Or
// ============================================================
#define TEST_UDMA_ATOMIC_OR_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_or_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_OR_FUNC);

#define TEST_UDMA_ATOMIC_OR_HOST(NAME, TYPE)                                                                           \
    static void test_udma_atomic_or_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,                  \
                                                  uint32_t rank_size)                                                  \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = sizeof(TYPE);                                                                               \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = static_cast<TYPE>(0x0F);                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_or_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                                     \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(0xFF));                                                                  \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_OR_HOST);


#define TEST_UDMA_ATOMIC_OR_MEM(NAME, TYPE)                                                                            \
    void test_udma_atomic_or_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                           \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_or_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);                     \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_OR_MEM);


// ============================================================
// Atomic Fetch Xor
// ============================================================
#define TEST_UDMA_ATOMIC_FETCH_XOR_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_fetch_xor_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_XOR_FUNC);

#define TEST_UDMA_ATOMIC_FETCH_XOR_HOST(NAME, TYPE)                                                                    \
    static void test_udma_atomic_fetch_xor_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,           \
                                                          uint32_t rank_size)                                          \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = 2 * sizeof(TYPE);                                                                           \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = static_cast<TYPE>(0xAA);                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_fetch_xor_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                              \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(0x55));                                                                  \
        ASSERT_EQ(xHost[1], static_cast<TYPE>(0xAA));                                                                  \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_XOR_HOST);

#define TEST_UDMA_ATOMIC_FETCH_XOR_MEM(NAME, TYPE)                                                                     \
    void test_udma_atomic_fetch_xor_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                    \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_fetch_xor_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);              \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_XOR_MEM);


// ============================================================
// Atomic Xor
// ============================================================
#define TEST_UDMA_ATOMIC_XOR_FUNC(NAME, TYPE) \
    extern void test_udma_atomic_xor_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_XOR_FUNC);

#define TEST_UDMA_ATOMIC_XOR_HOST(NAME, TYPE)                                                                          \
    static void test_udma_atomic_xor_##NAME##_host(aclrtStream stream, uint8_t *gva, uint32_t rank_id,                 \
                                                   uint32_t rank_size)                                                 \
    {                                                                                                                  \
        TYPE *xHost;                                                                                                   \
        size_t totalSize = sizeof(TYPE);                                                                               \
        ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);                                                   \
        xHost[0] = static_cast<TYPE>(0xAA);                                                                            \
        uint8_t *ptr = (uint8_t *)aclshmem_malloc(totalSize);                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                        \
        test_udma_atomic_xor_##NAME##_do(1, stream, (ptr), util_get_ffts_config());                                    \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                  \
        std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                             \
        std::cout << p_name;                                                                                           \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                        \
        ASSERT_EQ(xHost[0], static_cast<TYPE>(0x55));                                                                  \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                            \
        aclshmem_free(ptr);                                                                                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_XOR_HOST);

#define TEST_UDMA_ATOMIC_XOR_MEM(NAME, TYPE)                                                                           \
    void test_udma_atomic_xor_##NAME##_mem(int rank_id, int n_ranks, uint64_t local_mem_size)                          \
    {                                                                                                                  \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                                  \
        aclrtStream stream;                                                                                            \
        auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
        if (status != 0) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        ASSERT_NE(stream, nullptr);                                                                                    \
        test_udma_atomic_xor_##NAME##_host(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);                    \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                   \
        test_finalize(stream, device_id);                                                                              \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_XOR_MEM);


#define TEST_UDMA_ATOMIC_ADD_API(NAME, TYPE)                                                                           \
    TEST(TestMemApi, TestShmemUDMAAtomicAdd##NAME##Mem)                                                                \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_add_##NAME##_mem, local_mem_size, processCount);*/                          \
    }
UDMA_ATOMIC_ADD_FUNC_TYPE(TEST_UDMA_ATOMIC_ADD_API);

#define TEST_UDMA_ATOMIC_FETCH_ADD_API(NAME, TYPE)                                                                     \
    TEST(TestMemApi, TestShmemUDMAAtomicFetchAdd##NAME##Mem)                                                           \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_fetch_add_##NAME##_mem, local_mem_size, processCount);*/                    \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_ADD_API);

#define TEST_UDMA_ATOMIC_COMPARE_SWAP_API(NAME, TYPE)                                                                  \
    TEST(TestMemApi, TestShmemUDMAAtomicCompareSwap##NAME##Mem)                                                        \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_compare_swap_##NAME##_mem, local_mem_size, processCount);*/                 \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_COMPARE_SWAP_API);

#define TEST_UDMA_ATOMIC_FETCH_API(NAME, TYPE)                                                                          \
    TEST(TestMemApi, TestShmemUDMAAtomicFetch##NAME##Mem)                                                              \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_fetch_##NAME##_mem, local_mem_size, processCount);*/                        \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_API);

#define TEST_UDMA_ATOMIC_SET_API(NAME, TYPE)                                                                            \
    TEST(TestMemApi, TestShmemUDMAAtomicSet##NAME##Mem)                                                                 \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_set_##NAME##_mem, local_mem_size, processCount);*/                          \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_SET_API);

#define TEST_UDMA_ATOMIC_SWAP_API(NAME, TYPE)                                                                           \
    TEST(TestMemApi, TestShmemUDMAAtomicSwap##NAME##Mem)                                                               \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_swap_##NAME##_mem, local_mem_size, processCount);*/                         \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_SWAP_API);

#define TEST_UDMA_ATOMIC_FETCH_INC_API(NAME, TYPE)                                                                      \
    TEST(TestMemApi, TestShmemUDMAAtomicFetchInc##NAME##Mem)                                                           \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_fetch_inc_##NAME##_mem, local_mem_size, processCount);*/                    \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_INC_API);

#define TEST_UDMA_ATOMIC_INC_API(NAME, TYPE)                                                                            \
    TEST(TestMemApi, TestShmemUDMAAtomicInc##NAME##Mem)                                                                 \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_inc_##NAME##_mem, local_mem_size, processCount);*/                          \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_INC_API);

#define TEST_UDMA_ATOMIC_FETCH_AND_API(NAME, TYPE)                                                                      \
    TEST(TestMemApi, TestShmemUDMAAtomicFetchAnd##NAME##Mem)                                                           \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_fetch_and_##NAME##_mem, local_mem_size, processCount);*/                    \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_AND_API);

#define TEST_UDMA_ATOMIC_AND_API(NAME, TYPE)                                                                            \
    TEST(TestMemApi, TestShmemUDMAAtomicAnd##NAME##Mem)                                                                 \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_and_##NAME##_mem, local_mem_size, processCount);*/                          \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_AND_API);

#define TEST_UDMA_ATOMIC_FETCH_OR_API(NAME, TYPE)                                                                       \
    TEST(TestMemApi, TestShmemUDMAAtomicFetchOr##NAME##Mem)                                                            \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_fetch_or_##NAME##_mem, local_mem_size, processCount);*/                     \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_OR_API);

#define TEST_UDMA_ATOMIC_OR_API(NAME, TYPE)                                                                             \
    TEST(TestMemApi, TestShmemUDMAAtomicOr##NAME##Mem)                                                                  \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_or_##NAME##_mem, local_mem_size, processCount);*/                            \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_OR_API);

#define TEST_UDMA_ATOMIC_FETCH_XOR_API(NAME, TYPE)                                                                      \
    TEST(TestMemApi, TestShmemUDMAAtomicFetchXor##NAME##Mem)                                                           \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_fetch_xor_##NAME##_mem, local_mem_size, processCount);*/                    \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_FETCH_XOR_API);

#define TEST_UDMA_ATOMIC_XOR_API(NAME, TYPE)                                                                            \
    TEST(TestMemApi, TestShmemUDMAAtomicXor##NAME##Mem)                                                                 \
    {                                                                                                                  \
        const int processCount = test_gnpu_num;                                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                              \
        /*test_mutil_task(test_udma_atomic_xor_##NAME##_mem, local_mem_size, processCount);*/                          \
    }
UDMA_ATOMIC_FUNC_TYPE(TEST_UDMA_ATOMIC_XOR_API);