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
 * @file atomic_logic_rdma_host_test.cpp
 * @brief Unit test host code for RDMA atomic logic operations:
 *        - atomic_and
 *        - atomic_or
 *        - atomic_xor
 *        - atomic_fetch_and
 *        - atomic_fetch_or
 *        - atomic_fetch_xor
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
#include "atomic.h"

/*****************************************************************************
 *                    atomic_and test (RDMA)                                  *
 *****************************************************************************/
#ifdef ACLSHMEMI_RDMA_K_BACKEND_XSCALE
#define TEST_RDMA_ATOMIC_AND_FUNC(NAME, TYPE) \
    extern void test_rdma_atomic_and_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config)
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_AND_FUNC);

#define TEST_ACLSHMEM_RDMA_ATOMIC_AND_HOST(NAME, TYPE)                                                       \
    static void test_rdma_atomic_and_##NAME##_host(aclrtStream stream, uint32_t pe_id, uint32_t pe_size)     \
    {                                                                                                        \
        size_t messageSize = 64;                                                                             \
        TYPE* xHost;                                                                                         \
        size_t totalSize = messageSize * pe_size;                                                            \
                                                                                                             \
        ASSERT_EQ(aclrtMallocHost((void**)(&xHost), totalSize), 0);                                          \
        for (uint32_t i = 0; i < totalSize / sizeof(TYPE); i++) {                                            \
            xHost[i] = ~(TYPE)0;                                                                             \
        }                                                                                                    \
        xHost[pe_id * messageSize / sizeof(TYPE)] = pe_id + 1;                                               \
                                                                                                             \
        TYPE* ptr = (TYPE*)aclshmem_malloc(totalSize);                                                       \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);              \
                                                                                                             \
        uint32_t block_dim = 1;                                                                              \
        test_rdma_atomic_and_##NAME##_do(block_dim, stream, (TYPE*)ptr, util_get_ffts_config());             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                        \
                                                                                                             \
        std::string p_name = "[Process " + std::to_string(pe_id) + "] ";                                     \
        std::cout << p_name;                                                                                 \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);              \
        for (uint32_t i = 0; i < pe_size; i++) {                                                             \
            if (i == pe_id) {                                                                                \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(pe_id + 1));              \
            } else {                                                                                         \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(~((TYPE)((1LLU << i))))); \
            }                                                                                                \
        }                                                                                                    \
        aclshmem_free(ptr);                                                                                  \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                  \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_AND_HOST);

#define TEST_ACLSHMEM_RDMA_ATOMIC_AND(NAME, TYPE)                                                  \
    void test_aclshmem_rdma_atomic_and_##NAME##_mem(int pe_id, int n_pes, uint64_t local_mem_size) \
    {                                                                                              \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                                \
        aclrtStream stream;                                                                        \
        test_rdma_init(pe_id, n_pes, local_mem_size, &stream);                                     \
        ASSERT_NE(stream, nullptr);                                                                \
        test_rdma_atomic_and_##NAME##_host(stream, pe_id, n_pes);                                  \
        std::cout << "[TEST] begin to exit...... pe_id: " << pe_id << std::endl;                   \
        test_finalize(stream, device_id);                                                          \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_AND);

#define TEST_RDMA_ATOMIC_AND_API(NAME, TYPE)                                                           \
    TEST(TestMemApi, TestShmemRdmaAtomicAnd##NAME##Mem)                                                \
    {                                                                                                  \
        const int processCount = test_gnpu_num;                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                                \
        if (is_hardware_atomic_supported()) {                                                          \
            test_mutil_task(test_aclshmem_rdma_atomic_and_##NAME##_mem, local_mem_size, processCount); \
        }                                                                                              \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_AND_API);

/*****************************************************************************
 *                    atomic_fetch_and test (RDMA)                            *
 *****************************************************************************/
#define TEST_RDMA_ATOMIC_FETCH_AND_FUNC(NAME, TYPE)     \
    extern void test_rdma_atomic_fetch_and_##NAME##_do( \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t pe_size, uint64_t config)
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_FETCH_AND_FUNC);

#define TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_AND_HOST(NAME, TYPE)                                                       \
    static void test_rdma_atomic_fetch_and_##NAME##_host(aclrtStream stream, uint32_t pe_id, uint32_t pe_size)     \
    {                                                                                                              \
        size_t messageSize = 64;                                                                                   \
        TYPE* xHost;                                                                                               \
        size_t totalSize = messageSize * pe_size;                                                                  \
                                                                                                                   \
        ASSERT_EQ(aclrtMallocHost((void**)(&xHost), totalSize), 0);                                                \
        for (uint32_t i = 0; i < totalSize / sizeof(TYPE); i++) {                                                  \
            xHost[i] = ~(TYPE)0;                                                                                   \
        }                                                                                                          \
        xHost[pe_id * messageSize / sizeof(TYPE)] = pe_id + 1;                                                     \
                                                                                                                   \
        TYPE* ptr = (TYPE*)aclshmem_malloc(totalSize);                                                             \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                    \
                                                                                                                   \
        uint32_t block_dim = 1;                                                                                    \
        int* error_flag_dev;                                                                                       \
        int* error_flag_host;                                                                                      \
        ASSERT_EQ(aclrtMalloc((void**)(&error_flag_dev), 512, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                     \
        ASSERT_EQ(aclrtMallocHost((void**)(&error_flag_host), sizeof(int)), 0);                                    \
        *error_flag_host = 0;                                                                                      \
        ASSERT_EQ(                                                                                                 \
            aclrtMemcpy(error_flag_dev, sizeof(int), error_flag_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE), 0); \
                                                                                                                   \
        test_rdma_atomic_fetch_and_##NAME##_do(                                                                    \
            block_dim, stream, (TYPE*)ptr, error_flag_dev, pe_size, util_get_ffts_config());                       \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                              \
                                                                                                                   \
        std::string p_name = "[Process " + std::to_string(pe_id) + "] ";                                           \
        std::cout << p_name;                                                                                       \
        ASSERT_EQ(                                                                                                 \
            aclrtMemcpy(error_flag_host, sizeof(int), error_flag_dev, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST), 0); \
        ASSERT_EQ(*error_flag_host, 0);                                                                            \
        aclrtFree(error_flag_dev);                                                                                 \
        aclrtFreeHost(error_flag_host);                                                                            \
                                                                                                                   \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                    \
        for (uint32_t i = 0; i < pe_size; i++) {                                                                   \
            if (i == pe_id) {                                                                                      \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(pe_id + 1));                    \
            } else {                                                                                               \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(~((TYPE)((1LLU << i)))));       \
            }                                                                                                      \
        }                                                                                                          \
        aclshmem_free(ptr);                                                                                        \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                        \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_AND_HOST);

#define TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_AND(NAME, TYPE)                                                  \
    void test_aclshmem_rdma_atomic_fetch_and_##NAME##_mem(int pe_id, int n_pes, uint64_t local_mem_size) \
    {                                                                                                    \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                                      \
        aclrtStream stream;                                                                              \
        test_rdma_init(pe_id, n_pes, local_mem_size, &stream);                                           \
        ASSERT_NE(stream, nullptr);                                                                      \
        test_rdma_atomic_fetch_and_##NAME##_host(stream, pe_id, n_pes);                                  \
        std::cout << "[TEST] begin to exit...... pe_id: " << pe_id << std::endl;                         \
        test_finalize(stream, device_id);                                                                \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_AND);

#define TEST_RDMA_ATOMIC_FETCH_AND_API(NAME, TYPE)                                                           \
    TEST(TestMemApi, TestShmemRdmaAtomicFetchAnd##NAME##Mem)                                                 \
    {                                                                                                        \
        const int processCount = test_gnpu_num;                                                              \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                                      \
        if (is_hardware_atomic_supported()) {                                                                \
            test_mutil_task(test_aclshmem_rdma_atomic_fetch_and_##NAME##_mem, local_mem_size, processCount); \
        }                                                                                                    \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_FETCH_AND_API);

/*****************************************************************************
 *                    atomic_or test (RDMA)                                   *
 *****************************************************************************/
#define TEST_RDMA_ATOMIC_OR_FUNC(NAME, TYPE) \
    extern void test_rdma_atomic_or_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config)
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_OR_FUNC);

#define TEST_ACLSHMEM_RDMA_ATOMIC_OR_HOST(NAME, TYPE)                                                   \
    static void test_rdma_atomic_or_##NAME##_host(aclrtStream stream, uint32_t pe_id, uint32_t pe_size) \
    {                                                                                                   \
        size_t messageSize = 64;                                                                        \
        TYPE* xHost;                                                                                    \
        size_t totalSize = messageSize * pe_size;                                                       \
                                                                                                        \
        ASSERT_EQ(aclrtMallocHost((void**)(&xHost), totalSize), 0);                                     \
        for (uint32_t i = 0; i < totalSize / sizeof(TYPE); i++) {                                       \
            xHost[i] = 0;                                                                               \
        }                                                                                               \
        xHost[pe_id * messageSize / sizeof(TYPE)] = pe_id + 1;                                          \
                                                                                                        \
        TYPE* ptr = (TYPE*)aclshmem_malloc(totalSize);                                                  \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);         \
                                                                                                        \
        uint32_t block_dim = 1;                                                                         \
        test_rdma_atomic_or_##NAME##_do(block_dim, stream, (TYPE*)ptr, util_get_ffts_config());         \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                   \
                                                                                                        \
        std::string p_name = "[Process " + std::to_string(pe_id) + "] ";                                \
        std::cout << p_name;                                                                            \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);         \
        for (uint32_t i = 0; i < pe_size; i++) {                                                        \
            if (i == pe_id) {                                                                           \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(pe_id + 1));         \
            } else {                                                                                    \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>((1LLU << i)));       \
            }                                                                                           \
        }                                                                                               \
        aclshmem_free(ptr);                                                                             \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                             \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_OR_HOST);

#define TEST_ACLSHMEM_RDMA_ATOMIC_OR(NAME, TYPE)                                                  \
    void test_aclshmem_rdma_atomic_or_##NAME##_mem(int pe_id, int n_pes, uint64_t local_mem_size) \
    {                                                                                             \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                               \
        aclrtStream stream;                                                                       \
        test_rdma_init(pe_id, n_pes, local_mem_size, &stream);                                    \
        ASSERT_NE(stream, nullptr);                                                               \
        test_rdma_atomic_or_##NAME##_host(stream, pe_id, n_pes);                                  \
        std::cout << "[TEST] begin to exit...... pe_id: " << pe_id << std::endl;                  \
        test_finalize(stream, device_id);                                                         \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_OR);

#define TEST_RDMA_ATOMIC_OR_API(NAME, TYPE)                                                           \
    TEST(TestMemApi, TestShmemRdmaAtomicOr##NAME##Mem)                                                \
    {                                                                                                 \
        const int processCount = test_gnpu_num;                                                       \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                               \
        if (is_hardware_atomic_supported()) {                                                         \
            test_mutil_task(test_aclshmem_rdma_atomic_or_##NAME##_mem, local_mem_size, processCount); \
        }                                                                                             \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_OR_API);

/*****************************************************************************
 *                    atomic_fetch_or test (RDMA)                             *
 *****************************************************************************/
#define TEST_RDMA_ATOMIC_FETCH_OR_FUNC(NAME, TYPE)     \
    extern void test_rdma_atomic_fetch_or_##NAME##_do( \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t pe_size, uint64_t config)
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_FETCH_OR_FUNC);

#define TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_OR_HOST(NAME, TYPE)                                                        \
    static void test_rdma_atomic_fetch_or_##NAME##_host(aclrtStream stream, uint32_t pe_id, uint32_t pe_size)      \
    {                                                                                                              \
        size_t messageSize = 64;                                                                                   \
        TYPE* xHost;                                                                                               \
        size_t totalSize = messageSize * pe_size;                                                                  \
                                                                                                                   \
        ASSERT_EQ(aclrtMallocHost((void**)(&xHost), totalSize), 0);                                                \
        for (uint32_t i = 0; i < totalSize / sizeof(TYPE); i++) {                                                  \
            xHost[i] = 0;                                                                                          \
        }                                                                                                          \
        xHost[pe_id * messageSize / sizeof(TYPE)] = pe_id + 1;                                                     \
                                                                                                                   \
        TYPE* ptr = (TYPE*)aclshmem_malloc(totalSize);                                                             \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                    \
                                                                                                                   \
        uint32_t block_dim = 1;                                                                                    \
        int* error_flag_dev;                                                                                       \
        int* error_flag_host;                                                                                      \
        ASSERT_EQ(aclrtMalloc((void**)(&error_flag_dev), 512, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                     \
        ASSERT_EQ(aclrtMallocHost((void**)(&error_flag_host), sizeof(int)), 0);                                    \
        *error_flag_host = 0;                                                                                      \
        ASSERT_EQ(                                                                                                 \
            aclrtMemcpy(error_flag_dev, sizeof(int), error_flag_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE), 0); \
                                                                                                                   \
        test_rdma_atomic_fetch_or_##NAME##_do(                                                                     \
            block_dim, stream, (TYPE*)ptr, error_flag_dev, pe_size, util_get_ffts_config());                       \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                              \
                                                                                                                   \
        std::string p_name = "[Process " + std::to_string(pe_id) + "] ";                                           \
        std::cout << p_name;                                                                                       \
        ASSERT_EQ(                                                                                                 \
            aclrtMemcpy(error_flag_host, sizeof(int), error_flag_dev, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST), 0); \
        ASSERT_EQ(*error_flag_host, 0);                                                                            \
        aclrtFree(error_flag_dev);                                                                                 \
        aclrtFreeHost(error_flag_host);                                                                            \
                                                                                                                   \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                    \
        for (uint32_t i = 0; i < pe_size; i++) {                                                                   \
            if (i == pe_id) {                                                                                      \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(pe_id + 1));                    \
            } else {                                                                                               \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>((1LLU << i)));                  \
            }                                                                                                      \
        }                                                                                                          \
        aclshmem_free(ptr);                                                                                        \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                        \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_OR_HOST);

#define TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_OR(NAME, TYPE)                                                  \
    void test_aclshmem_rdma_atomic_fetch_or_##NAME##_mem(int pe_id, int n_pes, uint64_t local_mem_size) \
    {                                                                                                   \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                                     \
        aclrtStream stream;                                                                             \
        test_rdma_init(pe_id, n_pes, local_mem_size, &stream);                                          \
        ASSERT_NE(stream, nullptr);                                                                     \
        test_rdma_atomic_fetch_or_##NAME##_host(stream, pe_id, n_pes);                                  \
        std::cout << "[TEST] begin to exit...... pe_id: " << pe_id << std::endl;                        \
        test_finalize(stream, device_id);                                                               \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_OR);

#define TEST_RDMA_ATOMIC_FETCH_OR_API(NAME, TYPE)                                                           \
    TEST(TestMemApi, TestShmemRdmaAtomicFetchOr##NAME##Mem)                                                 \
    {                                                                                                       \
        const int processCount = test_gnpu_num;                                                             \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                                     \
        if (is_hardware_atomic_supported()) {                                                               \
            test_mutil_task(test_aclshmem_rdma_atomic_fetch_or_##NAME##_mem, local_mem_size, processCount); \
        }                                                                                                   \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_FETCH_OR_API);

/*****************************************************************************
 *                    atomic_xor test (RDMA)                                  *
 *****************************************************************************/
#define TEST_RDMA_ATOMIC_XOR_FUNC(NAME, TYPE) \
    extern void test_rdma_atomic_xor_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config)
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_XOR_FUNC);

#define TEST_ACLSHMEM_RDMA_ATOMIC_XOR_HOST(NAME, TYPE)                                                       \
    static void test_rdma_atomic_xor_##NAME##_host(aclrtStream stream, uint32_t pe_id, uint32_t pe_size)     \
    {                                                                                                        \
        size_t messageSize = 64;                                                                             \
        TYPE* xHost;                                                                                         \
        size_t totalSize = messageSize * pe_size;                                                            \
                                                                                                             \
        ASSERT_EQ(aclrtMallocHost((void**)(&xHost), totalSize), 0);                                          \
        for (uint32_t i = 0; i < totalSize / sizeof(TYPE); i++) {                                            \
            xHost[i] = ~(TYPE)0;                                                                             \
        }                                                                                                    \
        xHost[pe_id * messageSize / sizeof(TYPE)] = pe_id + 1;                                               \
                                                                                                             \
        TYPE* ptr = (TYPE*)aclshmem_malloc(totalSize);                                                       \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);              \
                                                                                                             \
        uint32_t block_dim = 1;                                                                              \
        test_rdma_atomic_xor_##NAME##_do(block_dim, stream, (TYPE*)ptr, util_get_ffts_config());             \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                        \
                                                                                                             \
        std::string p_name = "[Process " + std::to_string(pe_id) + "] ";                                     \
        std::cout << p_name;                                                                                 \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);              \
        for (uint32_t i = 0; i < pe_size; i++) {                                                             \
            if (i == pe_id) {                                                                                \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(pe_id + 1));              \
            } else {                                                                                         \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(~((TYPE)((1LLU << i))))); \
            }                                                                                                \
        }                                                                                                    \
        aclshmem_free(ptr);                                                                                  \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                  \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_XOR_HOST);

#define TEST_ACLSHMEM_RDMA_ATOMIC_XOR(NAME, TYPE)                                                  \
    void test_aclshmem_rdma_atomic_xor_##NAME##_mem(int pe_id, int n_pes, uint64_t local_mem_size) \
    {                                                                                              \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                                \
        aclrtStream stream;                                                                        \
        test_rdma_init(pe_id, n_pes, local_mem_size, &stream);                                     \
        ASSERT_NE(stream, nullptr);                                                                \
        test_rdma_atomic_xor_##NAME##_host(stream, pe_id, n_pes);                                  \
        std::cout << "[TEST] begin to exit...... pe_id: " << pe_id << std::endl;                   \
        test_finalize(stream, device_id);                                                          \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_XOR);

#define TEST_RDMA_ATOMIC_XOR_API(NAME, TYPE)                                                           \
    TEST(TestMemApi, TestShmemRdmaAtomicXor##NAME##Mem)                                                \
    {                                                                                                  \
        const int processCount = test_gnpu_num;                                                        \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                                \
        if (is_hardware_atomic_supported()) {                                                          \
            test_mutil_task(test_aclshmem_rdma_atomic_xor_##NAME##_mem, local_mem_size, processCount); \
        }                                                                                              \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_XOR_API);

/*****************************************************************************
 *                    atomic_fetch_xor test (RDMA)                            *
 *****************************************************************************/
#define TEST_RDMA_ATOMIC_FETCH_XOR_FUNC(NAME, TYPE)     \
    extern void test_rdma_atomic_fetch_xor_##NAME##_do( \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t pe_size, uint64_t config)
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_FETCH_XOR_FUNC);

#define TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_XOR_HOST(NAME, TYPE)                                                       \
    static void test_rdma_atomic_fetch_xor_##NAME##_host(aclrtStream stream, uint32_t pe_id, uint32_t pe_size)     \
    {                                                                                                              \
        size_t messageSize = 64;                                                                                   \
        TYPE* xHost;                                                                                               \
        size_t totalSize = messageSize * pe_size;                                                                  \
                                                                                                                   \
        ASSERT_EQ(aclrtMallocHost((void**)(&xHost), totalSize), 0);                                                \
        for (uint32_t i = 0; i < totalSize / sizeof(TYPE); i++) {                                                  \
            xHost[i] = ~(TYPE)0;                                                                                   \
        }                                                                                                          \
        xHost[pe_id * messageSize / sizeof(TYPE)] = pe_id + 1;                                                     \
                                                                                                                   \
        TYPE* ptr = (TYPE*)aclshmem_malloc(totalSize);                                                             \
        ASSERT_EQ(aclrtMemcpy(ptr, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);                    \
                                                                                                                   \
        uint32_t block_dim = 1;                                                                                    \
        /* Allocate error_flag buffer on device */                                                                 \
        int* error_flag_dev;                                                                                       \
        int* error_flag_host;                                                                                      \
        ASSERT_EQ(aclrtMalloc((void**)(&error_flag_dev), 512, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                     \
        ASSERT_EQ(aclrtMallocHost((void**)(&error_flag_host), sizeof(int)), 0);                                    \
        *error_flag_host = 0;                                                                                      \
        ASSERT_EQ(                                                                                                 \
            aclrtMemcpy(error_flag_dev, sizeof(int), error_flag_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE), 0); \
                                                                                                                   \
        test_rdma_atomic_fetch_xor_##NAME##_do(                                                                    \
            block_dim, stream, (TYPE*)ptr, error_flag_dev, pe_size, util_get_ffts_config());                       \
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                              \
                                                                                                                   \
        std::string p_name = "[Process " + std::to_string(pe_id) + "] ";                                           \
        std::cout << p_name;                                                                                       \
        ASSERT_EQ(                                                                                                 \
            aclrtMemcpy(error_flag_host, sizeof(int), error_flag_dev, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST), 0); \
        ASSERT_EQ(*error_flag_host, 0);                                                                            \
        aclrtFree(error_flag_dev);                                                                                 \
        aclrtFreeHost(error_flag_host);                                                                            \
                                                                                                                   \
        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, ptr, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);                    \
        for (uint32_t i = 0; i < pe_size; i++) {                                                                   \
            if (i == pe_id) {                                                                                      \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(pe_id + 1));                    \
            } else {                                                                                               \
                ASSERT_EQ(xHost[i * messageSize / sizeof(TYPE)], static_cast<TYPE>(~((TYPE)((1LLU << i)))));       \
            }                                                                                                      \
        }                                                                                                          \
        aclshmem_free(ptr);                                                                                        \
        ASSERT_EQ(aclrtFreeHost(xHost), 0);                                                                        \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_XOR_HOST);

#define TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_XOR(NAME, TYPE)                                                  \
    void test_aclshmem_rdma_atomic_fetch_xor_##NAME##_mem(int pe_id, int n_pes, uint64_t local_mem_size) \
    {                                                                                                    \
        int32_t device_id = pe_id % test_gnpu_num + test_first_npu;                                      \
        aclrtStream stream;                                                                              \
        test_rdma_init(pe_id, n_pes, local_mem_size, &stream);                                           \
        ASSERT_NE(stream, nullptr);                                                                      \
        test_rdma_atomic_fetch_xor_##NAME##_host(stream, pe_id, n_pes);                                  \
        std::cout << "[TEST] begin to exit...... pe_id: " << pe_id << std::endl;                         \
        test_finalize(stream, device_id);                                                                \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_ACLSHMEM_RDMA_ATOMIC_FETCH_XOR);

#define TEST_RDMA_ATOMIC_FETCH_XOR_API(NAME, TYPE)                                                           \
    TEST(TestMemApi, TestShmemRdmaAtomicFetchXor##NAME##Mem)                                                 \
    {                                                                                                        \
        const int processCount = test_gnpu_num;                                                              \
        uint64_t local_mem_size = 1024UL * 1024UL * 64;                                                      \
        if (is_hardware_atomic_supported()) {                                                                \
            test_mutil_task(test_aclshmem_rdma_atomic_fetch_xor_##NAME##_mem, local_mem_size, processCount); \
        }                                                                                                    \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(TEST_RDMA_ATOMIC_FETCH_XOR_API);
#endif // ACLSHMEMI_RDMA_K_BACKEND_XSCALE
