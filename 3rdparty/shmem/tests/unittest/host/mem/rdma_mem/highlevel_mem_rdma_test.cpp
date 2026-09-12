/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file highlevel_mem_rdma_test.cpp
 * @brief Combined unit tests for high-level RMA interfaces under RDMA/ROCE engine
 *
 * This file contains host-side combined test cases for testing PUT+GET API pairs:
 * - aclshmem_putmem + aclshmem_getmem: blocking, pointer interface
 * - aclshmem_putmem_nbi + aclshmem_getmem_nbi: non-blocking, pointer interface
 *
 * Multi-type tests for typed PUT/GET (using ACLSHMEM_MEM_PUT_GET_FUNC macro):
 * - aclshmem_##NAME##_put + aclshmem_##NAME##_get: typed blocking PUT+GET for all 11 types
 * - aclshmem_##NAME##_put_nbi + aclshmem_##NAME##_get_nbi: typed non-blocking PUT+GET for all 11 types
 * - aclshmem_##NAME##_put_nbi + aclshmem_##NAME##_get_nbi (GlobalTensor): typed non-blocking PUT+GET with GlobalTensor
 * for all 11 types
 * - Supported types: float, double, int8, int16, int32, int64, uint8, uint16, uint32, uint64, char
 */

#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <cstring>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"

// Device kernel functions (defined in highlevel_mem_rdma_test_kernel.cpp)
extern void test_rdma_getmem(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
extern void test_rdma_putmem(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
extern void test_rdma_getmem_nbi(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
extern void test_rdma_putmem_nbi(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);

// ============================================================================
// Combined tests: PUT + GET in single test
// ============================================================================

/**
 * Combined test for aclshmem_putmem + aclshmem_getmem (blocking, pointer interface)
 *
 * Test flow:
 * 1. Each PE initializes its own data region with (pe_id + 10)
 * 2. PUT test: Each PE calls aclshmem_putmem to send data to other peers, then verify
 * 3. GET test: Each PE calls aclshmem_getmem to fetch data from other peers, then verify
 */
static void test_rdma_put_get_mem_func(aclrtStream stream, uint8_t* gva, uint32_t pe_id, uint32_t pe_size)
{
    size_t messageSize = 64;
    uint32_t peOffset = 10;
    uint32_t* inHost;
    uint32_t* outHost;
    size_t totalSize = messageSize * pe_size;
    uint32_t block_dim = 1;

    // Allocate host memory
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);
    bzero(inHost, totalSize);

    // Initialize: each PE fills its own region with (pe_id + peOffset)
    for (uint32_t i = 0; i < messageSize / sizeof(uint32_t); i++) {
        inHost[i + pe_id * messageSize / sizeof(uint32_t)] = pe_id + peOffset;
    }

    // ========== PUT test: putmem ==========
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_rdma_putmem(block_dim, stream, gva, util_get_ffts_config());
    aclshmem_handle_t handle;
    handle.team_id = ACLSHMEM_TEAM_WORLD;
    aclshmemx_handle_wait(handle, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        uint32_t expected = i + peOffset;
        uint32_t actual = outHost[i * messageSize / sizeof(uint32_t)];
        ASSERT_EQ(actual, expected) << "PUT failed: PE " << pe_id << " at index " << i;
    }

    // ========== GET test: getmem ==========
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_rdma_getmem(block_dim, stream, gva, util_get_ffts_config());
    handle.team_id = ACLSHMEM_TEAM_WORLD;
    aclshmemx_handle_wait(handle, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        uint32_t expected = i + peOffset;
        uint32_t actual = outHost[i * messageSize / sizeof(uint32_t)];
        ASSERT_EQ(actual, expected) << "GET failed: PE " << pe_id << " at index " << i;
    }

    ASSERT_EQ(aclrtFreeHost(inHost), 0);
    ASSERT_EQ(aclrtFreeHost(outHost), 0);
}

void test_aclshmem_rdma_put_get_mem(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id = pe_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(pe_id, n_pes, local_mem_size, &stream);
    if (status != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    size_t messageSize = 64;
    size_t totalSize = messageSize * n_pes;
    void* ptr = aclshmem_malloc(totalSize);
    ASSERT_NE(ptr, nullptr);
    test_rdma_put_get_mem_func(stream, (uint8_t*)ptr, pe_id, n_pes);
    std::cout << "[TEST] putmem+getmem (RDMA) finished, pe_id: " << pe_id << std::endl;
    aclshmem_free(ptr);
    test_finalize(stream, device_id);
}

/**
 * Combined unit test for aclshmem_putmem + aclshmem_getmem under RDMA/ROCE engine
 */
TEST(TestHighlevelMemApi, TestShmemPutGetMemBlockingRDMA)
{
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_aclshmem_rdma_put_get_mem, local_mem_size, processCount);
}

/**
 * Combined test for aclshmem_putmem_nbi + aclshmem_getmem_nbi (non-blocking, pointer interface)
 *
 * Test flow:
 * 1. Each PE initializes its own data region with (pe_id + 10)
 * 2. PUT test: Each PE calls aclshmem_putmem_nbi to send data to other peers, then verify
 * 3. GET test: Each PE calls aclshmem_getmem_nbi to fetch data from other peers, then verify
 */
static void test_rdma_put_get_mem_nbi_func(aclrtStream stream, uint8_t* gva, uint32_t pe_id, uint32_t pe_size)
{
    size_t messageSize = 64;
    uint32_t peOffset = 10;
    uint32_t* inHost;
    uint32_t* outHost;
    size_t totalSize = messageSize * pe_size;
    uint32_t block_dim = 1;

    // Allocate host memory
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);
    bzero(inHost, totalSize);

    // Initialize: each PE fills its own region with (pe_id + peOffset)
    for (uint32_t i = 0; i < messageSize / sizeof(uint32_t); i++) {
        inHost[i + pe_id * messageSize / sizeof(uint32_t)] = pe_id + peOffset;
    }

    // ========== PUT test: putmem_nbi ==========
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_rdma_putmem_nbi(block_dim, stream, gva, util_get_ffts_config());
    aclshmem_handle_t handle;
    handle.team_id = ACLSHMEM_TEAM_WORLD;
    aclshmemx_handle_wait(handle, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        uint32_t expected = i + peOffset;
        uint32_t actual = outHost[i * messageSize / sizeof(uint32_t)];
        ASSERT_EQ(actual, expected) << "PUT failed: PE " << pe_id << " at index " << i;
    }

    // ========== GET test: getmem_nbi ==========
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_rdma_getmem_nbi(block_dim, stream, gva, util_get_ffts_config());
    handle.team_id = ACLSHMEM_TEAM_WORLD;
    aclshmemx_handle_wait(handle, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        uint32_t expected = i + peOffset;
        uint32_t actual = outHost[i * messageSize / sizeof(uint32_t)];
        ASSERT_EQ(actual, expected) << "GET failed: PE " << pe_id << " at index " << i;
    }

    ASSERT_EQ(aclrtFreeHost(inHost), 0);
    ASSERT_EQ(aclrtFreeHost(outHost), 0);
}

void test_aclshmem_rdma_put_get_mem_nbi(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id = pe_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(pe_id, n_pes, local_mem_size, &stream);
    if (status != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    size_t messageSize = 64;
    size_t totalSize = messageSize * n_pes;
    void* ptr = aclshmem_malloc(totalSize);
    ASSERT_NE(ptr, nullptr);
    test_rdma_put_get_mem_nbi_func(stream, (uint8_t*)ptr, pe_id, n_pes);
    std::cout << "[TEST] putmem_nbi+getmem_nbi (RDMA) finished, pe_id: " << pe_id << std::endl;
    aclshmem_free(ptr);
    test_finalize(stream, device_id);
}

/**
 * Combined unit test for aclshmem_putmem_nbi + aclshmem_getmem_nbi under RDMA/ROCE engine
 */
TEST(TestHighlevelMemApi, TestShmemPutGetMemNbiRDMA)
{
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_aclshmem_rdma_put_get_mem_nbi, local_mem_size, processCount);
}

// ============================================================================
// Section: Multi-type typed PUT/GET tests (using ACLSHMEM_MEM_PUT_GET_FUNC macro)
// ============================================================================

#include "unittest/utils/func_type.h"

// ============================================================================
// Section 1: Declare device kernel wrappers for typed PUT/GET
// ============================================================================
#define DECLARE_PUT_GET_WRAPPER(NAME, TYPE)                                                              \
    extern void test_rdma_##NAME##_put(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config); \
    extern void test_rdma_##NAME##_get(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)

ACLSHMEM_MEM_PUT_GET_FUNC(DECLARE_PUT_GET_WRAPPER);
#undef DECLARE_PUT_GET_WRAPPER

// ============================================================================
// Section 2: Define test logic functions for typed PUT+GET combined test
// ============================================================================
#define DEFINE_PUT_GET_TEST(NAME, TYPE)                                                                             \
    static void test_rdma_##NAME##_put_get_func(aclrtStream stream, uint8_t* gva, uint32_t pe_id, uint32_t pe_size) \
    {                                                                                                               \
        size_t messageSize = 64;                                                                                    \
        uint32_t peOffset = 10;                                                                                     \
        TYPE* inHost;                                                                                               \
        TYPE* outHost;                                                                                              \
        size_t totalSize = messageSize * pe_size;                                                                   \
        uint32_t block_dim = 1;                                                                                     \
                                                                                                                    \
        ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);                                \
        ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);                               \
        bzero(inHost, totalSize);                                                                                   \
                                                                                                                    \
        for (uint32_t i = 0; i < messageSize / sizeof(TYPE); i++) {                                                 \
            inHost[i + pe_id * messageSize / sizeof(TYPE)] = static_cast<TYPE>(pe_id + peOffset);                   \
        }                                                                                                           \
                                                                                                                    \
        aclshmem_handle_t handle;                                                                                   \
        handle.team_id = ACLSHMEM_TEAM_WORLD;                                                                       \
                                                                                                                    \
        /* ========== PUT test ========== */                                                                        \
        ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                    \
        aclshmemi_control_barrier_all();                                                                            \
        test_rdma_##NAME##_put(block_dim, stream, gva, util_get_ffts_config());                                     \
        aclshmemx_handle_wait(handle, stream);                                                                      \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                               \
        aclshmemi_control_barrier_all();                                                                            \
        ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                   \
        for (uint32_t i = 0; i < pe_size; i++) {                                                                    \
            TYPE expected = static_cast<TYPE>(i + peOffset);                                                        \
            TYPE actual = outHost[i * messageSize / sizeof(TYPE)];                                                  \
            ASSERT_EQ(actual, expected) << "PUT failed: PE " << pe_id << " at index " << i;                         \
        }                                                                                                           \
                                                                                                                    \
        /* ========== GET test ========== */                                                                        \
        ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                    \
        aclshmemi_control_barrier_all();                                                                            \
        test_rdma_##NAME##_get(block_dim, stream, gva, util_get_ffts_config());                                     \
        aclshmemx_handle_wait(handle, stream);                                                                      \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                               \
        aclshmemi_control_barrier_all();                                                                            \
        ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                   \
        for (uint32_t i = 0; i < pe_size; i++) {                                                                    \
            TYPE expected = static_cast<TYPE>(i + peOffset);                                                        \
            TYPE actual = outHost[i * messageSize / sizeof(TYPE)];                                                  \
            ASSERT_EQ(actual, expected) << "GET failed: PE " << pe_id << " at index " << i;                         \
        }                                                                                                           \
                                                                                                                    \
        ASSERT_EQ(aclrtFreeHost(inHost), 0);                                                                        \
        ASSERT_EQ(aclrtFreeHost(outHost), 0);                                                                       \
    }

ACLSHMEM_MEM_PUT_GET_FUNC(DEFINE_PUT_GET_TEST);
#undef DEFINE_PUT_GET_TEST

// ============================================================================
// Section 3: Define entry functions and TEST macros for typed PUT+GET
// ============================================================================
#define DEFINE_PUT_GET_ENTRY(NAME, TYPE)                                                                  \
    void test_aclshmem_rdma_##NAME##_put_get(int pe_id, int n_pes, uint64_t local_mem_size)               \
    {                                                                                                     \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                                       \
        aclrtStream stream;                                                                               \
        auto status = test_rdma_init(pe_id, n_pes, local_mem_size, &stream);                              \
        if (status != 0) {                                                                                \
            return;                                                                                       \
        }                                                                                                 \
        ASSERT_NE(stream, nullptr);                                                                       \
                                                                                                          \
        size_t messageSize = 64;                                                                          \
        size_t totalSize = messageSize * n_pes;                                                           \
        void* ptr = aclshmem_malloc(totalSize);                                                           \
        ASSERT_NE(ptr, nullptr);                                                                          \
        test_rdma_##NAME##_put_get_func(stream, (uint8_t*)ptr, pe_id, n_pes);                             \
        std::cout << "[TEST] " #NAME "_put+" #NAME "_get (RDMA) finished, pe_id: " << pe_id << std::endl; \
        aclshmem_free(ptr);                                                                               \
        test_finalize(stream, device_id);                                                                 \
    }                                                                                                     \
                                                                                                          \
    TEST(TestTypedPutGetRDMA, TestShmem##NAME##PutGetBlockingRDMA)                                        \
    {                                                                                                     \
        const int processCount = test_gnpu_num;                                                           \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                                   \
        test_mutil_task(test_aclshmem_rdma_##NAME##_put_get, local_mem_size, processCount);               \
    }

ACLSHMEM_MEM_PUT_GET_FUNC(DEFINE_PUT_GET_ENTRY);
#undef DEFINE_PUT_GET_ENTRY

// ============================================================================
// Section: Multi-type typed PUT_NBI/GET_NBI tests (using ACLSHMEM_MEM_PUT_GET_FUNC macro)
// ============================================================================

// ============================================================================
// Section 4: Declare device kernel wrappers for typed PUT_NBI/GET_NBI
// ============================================================================
#define DECLARE_PUT_GET_NBI_WRAPPER(NAME, TYPE)                                                              \
    extern void test_rdma_##NAME##_put_nbi(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config); \
    extern void test_rdma_##NAME##_get_nbi(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)

ACLSHMEM_MEM_PUT_GET_FUNC(DECLARE_PUT_GET_NBI_WRAPPER);
#undef DECLARE_PUT_GET_NBI_WRAPPER

// ============================================================================
// Section 5: Define test logic functions for typed PUT_NBI+GET_NBI combined test
// ============================================================================
#define DEFINE_PUT_GET_NBI_TEST(NAME, TYPE)                                                       \
    static void test_rdma_##NAME##_put_get_nbi_func(                                              \
        aclrtStream stream, uint8_t* gva, uint32_t pe_id, uint32_t pe_size)                       \
    {                                                                                             \
        size_t messageSize = 64;                                                                  \
        uint32_t peOffset = 10;                                                                   \
        TYPE* inHost;                                                                             \
        TYPE* outHost;                                                                            \
        size_t totalSize = messageSize * pe_size;                                                 \
        uint32_t block_dim = 1;                                                                   \
                                                                                                  \
        ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);              \
        ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);             \
        bzero(inHost, totalSize);                                                                 \
                                                                                                  \
        for (uint32_t i = 0; i < messageSize / sizeof(TYPE); i++) {                               \
            inHost[i + pe_id * messageSize / sizeof(TYPE)] = static_cast<TYPE>(pe_id + peOffset); \
        }                                                                                         \
                                                                                                  \
        aclshmem_handle_t handle;                                                                 \
        handle.team_id = ACLSHMEM_TEAM_WORLD;                                                     \
                                                                                                  \
        /* ========== PUT_NBI test ========== */                                                  \
        ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);  \
        aclshmemi_control_barrier_all();                                                          \
        test_rdma_##NAME##_put_nbi(block_dim, stream, gva, util_get_ffts_config());               \
        aclshmemx_handle_wait(handle, stream);                                                    \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                             \
        aclshmemi_control_barrier_all();                                                          \
        ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0); \
        for (uint32_t i = 0; i < pe_size; i++) {                                                  \
            TYPE expected = static_cast<TYPE>(i + peOffset);                                      \
            TYPE actual = outHost[i * messageSize / sizeof(TYPE)];                                \
            ASSERT_EQ(actual, expected) << "PUT_NBI failed: PE " << pe_id << " at index " << i;   \
        }                                                                                         \
                                                                                                  \
        /* ========== GET_NBI test ========== */                                                  \
        ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);  \
        aclshmemi_control_barrier_all();                                                          \
        test_rdma_##NAME##_get_nbi(block_dim, stream, gva, util_get_ffts_config());               \
        aclshmemx_handle_wait(handle, stream);                                                    \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                             \
        aclshmemi_control_barrier_all();                                                          \
        ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0); \
        for (uint32_t i = 0; i < pe_size; i++) {                                                  \
            TYPE expected = static_cast<TYPE>(i + peOffset);                                      \
            TYPE actual = outHost[i * messageSize / sizeof(TYPE)];                                \
            ASSERT_EQ(actual, expected) << "GET_NBI failed: PE " << pe_id << " at index " << i;   \
        }                                                                                         \
                                                                                                  \
        ASSERT_EQ(aclrtFreeHost(inHost), 0);                                                      \
        ASSERT_EQ(aclrtFreeHost(outHost), 0);                                                     \
    }

ACLSHMEM_MEM_PUT_GET_FUNC(DEFINE_PUT_GET_NBI_TEST);
#undef DEFINE_PUT_GET_NBI_TEST

// ============================================================================
// Section 6: Define entry functions and TEST macros for typed PUT_NBI+GET_NBI
// ============================================================================
#define DEFINE_PUT_GET_NBI_ENTRY(NAME, TYPE)                                                                      \
    void test_aclshmem_rdma_##NAME##_put_get_nbi(int pe_id, int n_pes, uint64_t local_mem_size)                   \
    {                                                                                                             \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                                               \
        aclrtStream stream;                                                                                       \
        auto status = test_rdma_init(pe_id, n_pes, local_mem_size, &stream);                                      \
        if (status != 0) {                                                                                        \
            return;                                                                                               \
        }                                                                                                         \
        ASSERT_NE(stream, nullptr);                                                                               \
                                                                                                                  \
        size_t messageSize = 64;                                                                                  \
        size_t totalSize = messageSize * n_pes;                                                                   \
        void* ptr = aclshmem_malloc(totalSize);                                                                   \
        ASSERT_NE(ptr, nullptr);                                                                                  \
        test_rdma_##NAME##_put_get_nbi_func(stream, (uint8_t*)ptr, pe_id, n_pes);                                 \
        std::cout << "[TEST] " #NAME "_put_nbi+" #NAME "_get_nbi (RDMA) finished, pe_id: " << pe_id << std::endl; \
        aclshmem_free(ptr);                                                                                       \
        test_finalize(stream, device_id);                                                                         \
    }                                                                                                             \
                                                                                                                  \
    TEST(TestTypedPutGetNbiRDMA, TestShmem##NAME##PutGetNbiRDMA)                                                  \
    {                                                                                                             \
        const int processCount = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                                           \
        test_mutil_task(test_aclshmem_rdma_##NAME##_put_get_nbi, local_mem_size, processCount);                   \
    }

ACLSHMEM_MEM_PUT_GET_FUNC(DEFINE_PUT_GET_NBI_ENTRY);
#undef DEFINE_PUT_GET_NBI_ENTRY

// ============================================================================
// Section: Multi-type typed PUT_NBI/GET_NBI with GlobalTensor tests
// ============================================================================

// ============================================================================
// Section 7: Declare device kernel wrappers for typed PUT_NBI/GET_NBI Tensor
// ============================================================================
#define DECLARE_PUT_GET_NBI_TENSOR_WRAPPER(NAME, TYPE)                                                              \
    extern void test_rdma_##NAME##_put_nbi_tensor(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config); \
    extern void test_rdma_##NAME##_get_nbi_tensor(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)

ACLSHMEM_MEM_PUT_GET_FUNC(DECLARE_PUT_GET_NBI_TENSOR_WRAPPER);
#undef DECLARE_PUT_GET_NBI_TENSOR_WRAPPER

// ============================================================================
// Section 8: Define test logic functions for typed PUT_NBI+GET_NBI Tensor test
// ============================================================================
#define DEFINE_PUT_GET_NBI_TENSOR_TEST(NAME, TYPE)                                                     \
    static void test_rdma_##NAME##_put_get_nbi_tensor_func(                                            \
        aclrtStream stream, uint8_t* gva, uint32_t pe_id, uint32_t pe_size)                            \
    {                                                                                                  \
        size_t messageSize = 64;                                                                       \
        uint32_t peOffset = 10;                                                                        \
        TYPE* inHost;                                                                                  \
        TYPE* outHost;                                                                                 \
        size_t totalSize = messageSize * pe_size;                                                      \
        uint32_t block_dim = 1;                                                                        \
                                                                                                       \
        ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);                   \
        ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);                  \
        bzero(inHost, totalSize);                                                                      \
                                                                                                       \
        for (uint32_t i = 0; i < messageSize / sizeof(TYPE); i++) {                                    \
            inHost[i + pe_id * messageSize / sizeof(TYPE)] = static_cast<TYPE>(pe_id + peOffset);      \
        }                                                                                              \
                                                                                                       \
        aclshmem_handle_t handle;                                                                      \
        handle.team_id = ACLSHMEM_TEAM_WORLD;                                                          \
                                                                                                       \
        /* ========== PUT_NBI Tensor test ========== */                                                \
        ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);       \
        aclshmemi_control_barrier_all();                                                               \
        test_rdma_##NAME##_put_nbi_tensor(block_dim, stream, gva, util_get_ffts_config());             \
        aclshmemx_handle_wait(handle, stream);                                                         \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                  \
        aclshmemi_control_barrier_all();                                                               \
        ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);      \
        for (uint32_t i = 0; i < pe_size; i++) {                                                       \
            TYPE expected = static_cast<TYPE>(i + peOffset);                                           \
            TYPE actual = outHost[i * messageSize / sizeof(TYPE)];                                     \
            ASSERT_EQ(actual, expected) << "PUT_NBI_Tensor failed: PE " << pe_id << " at index " << i; \
        }                                                                                              \
                                                                                                       \
        /* ========== GET_NBI Tensor test ========== */                                                \
        ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);       \
        aclshmemi_control_barrier_all();                                                               \
        test_rdma_##NAME##_get_nbi_tensor(block_dim, stream, gva, util_get_ffts_config());             \
        aclshmemx_handle_wait(handle, stream);                                                         \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                  \
        aclshmemi_control_barrier_all();                                                               \
        ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);      \
        for (uint32_t i = 0; i < pe_size; i++) {                                                       \
            TYPE expected = static_cast<TYPE>(i + peOffset);                                           \
            TYPE actual = outHost[i * messageSize / sizeof(TYPE)];                                     \
            ASSERT_EQ(actual, expected) << "GET_NBI_Tensor failed: PE " << pe_id << " at index " << i; \
        }                                                                                              \
                                                                                                       \
        ASSERT_EQ(aclrtFreeHost(inHost), 0);                                                           \
        ASSERT_EQ(aclrtFreeHost(outHost), 0);                                                          \
    }

ACLSHMEM_MEM_PUT_GET_FUNC(DEFINE_PUT_GET_NBI_TENSOR_TEST);
#undef DEFINE_PUT_GET_NBI_TENSOR_TEST

// ============================================================================
// Section 9: Define entry functions and TEST macros for typed PUT_NBI+GET_NBI Tensor
// ============================================================================
#define DEFINE_PUT_GET_NBI_TENSOR_ENTRY(NAME, TYPE)                                                               \
    void test_aclshmem_rdma_##NAME##_put_get_nbi_tensor(int pe_id, int n_pes, uint64_t local_mem_size)            \
    {                                                                                                             \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                                               \
        aclrtStream stream;                                                                                       \
        auto status = test_rdma_init(pe_id, n_pes, local_mem_size, &stream);                                      \
        if (status != 0) {                                                                                        \
            return;                                                                                               \
        }                                                                                                         \
        ASSERT_NE(stream, nullptr);                                                                               \
                                                                                                                  \
        size_t messageSize = 64;                                                                                  \
        size_t totalSize = messageSize * n_pes;                                                                   \
        void* ptr = aclshmem_malloc(totalSize);                                                                   \
        ASSERT_NE(ptr, nullptr);                                                                                  \
        test_rdma_##NAME##_put_get_nbi_tensor_func(stream, (uint8_t*)ptr, pe_id, n_pes);                          \
        std::cout << "[TEST] " #NAME "_put_nbi_tensor+" #NAME "_get_nbi_tensor (RDMA) finished, pe_id: " << pe_id \
                  << std::endl;                                                                                   \
        aclshmem_free(ptr);                                                                                       \
        test_finalize(stream, device_id);                                                                         \
    }                                                                                                             \
                                                                                                                  \
    TEST(TestTypedPutGetNbiTensorRDMA, TestShmem##NAME##PutGetNbiTensorRDMA)                                      \
    {                                                                                                             \
        const int processCount = test_gnpu_num;                                                                   \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                                           \
        test_mutil_task(test_aclshmem_rdma_##NAME##_put_get_nbi_tensor, local_mem_size, processCount);            \
    }

ACLSHMEM_MEM_PUT_GET_FUNC(DEFINE_PUT_GET_NBI_TENSOR_ENTRY);
#undef DEFINE_PUT_GET_NBI_TENSOR_ENTRY