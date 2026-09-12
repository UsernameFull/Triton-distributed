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
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmem.h"
#include "unittest_main_test.h"
#include "utils/func_type.h"

using namespace std;

static constexpr int elem_size = 16;

#define PUT_SIZE(BITS, TYPE)                                                                                        \
    static void host_test_##BITS##_putmem(uint32_t rank_id)                                                         \
    {                                                                                                               \
        size_t input_size = elem_size * sizeof(TYPE);                                                               \
        std::vector<TYPE> input(elem_size, rank_id);                                                                \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            input[i] = i + 1;                                                                                       \
        }                                                                                                           \
        std::vector<TYPE> output(elem_size, 0);                                                                     \
                                                                                                                    \
        void *dev_ptr;                                                                                              \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                \
        ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);        \
        void *ptr = aclshmem_malloc(1024);                                                                          \
                                                                                                                    \
        aclshmem_put##BITS(ptr, dev_ptr, elem_size, rank_id);                                                       \
                                                                                                                    \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                          \
        sleep(2);                                                                                                   \
                                                                                                                    \
        ASSERT_EQ(aclrtMemcpy(output.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);           \
        int32_t flag = 0;                                                                                           \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            if (output[i] != input[i]) {                                                                            \
                flag = 1;                                                                                           \
            }                                                                                                       \
        }                                                                                                           \
        ASSERT_EQ(flag, 0);                                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(PUT_SIZE)
#undef PUT_SIZE

#define PUT_SIZE(BITS, TYPE)                                                                                        \
    void host_test_aclshmem_##BITS##_putmem(int rank_id, int n_ranks, uint64_t local_mem_size)                      \
    {                                                                                                               \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                               \
        aclrtStream stream;                                                                                         \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                                       \
        ASSERT_NE(stream, nullptr);                                                                                 \
        host_test_##BITS##_putmem(rank_id);                                                                         \
                                                                                                                    \
        test_finalize(stream, device_id);                                                                           \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(PUT_SIZE)
#undef PUT_SIZE

#define PUT_SIZE(BITS, TYPE)                                                                                        \
    TEST(TestMemHostApi, TestShmemPutSize##BITS##Mem)                                                               \
    {                                                                                                               \
        const int process_count = test_gnpu_num;                                                                    \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                           \
        test_mutil_task(                                                                                            \
            [](int rank_id, int n_rank, uint64_t local_memsize) {                                                   \
                host_test_aclshmem_##BITS##_putmem(rank_id, n_rank, local_memsize);                                 \
            },                                                                                                      \
            local_mem_size, process_count);                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(PUT_SIZE)
#undef PUT_SIZE


#define PUT_SIZE_NBI(BITS, TYPE)                                                                                    \
    static void host_test_##BITS##_putmem_nbi(uint32_t rank_id)                                                     \
    {                                                                                                               \
        size_t input_size = elem_size * sizeof(TYPE);                                                               \
        std::vector<TYPE> input(elem_size, rank_id);                                                                \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            input[i] = i + 1;                                                                                       \
        }                                                                                                           \
        std::vector<TYPE> output(elem_size, 0);                                                                     \
                                                                                                                    \
        void *dev_ptr;                                                                                              \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                \
        ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);        \
        void *ptr = aclshmem_malloc(1024);                                                                          \
                                                                                                                    \
        aclshmem_put##BITS##_nbi(ptr, dev_ptr, elem_size, rank_id);                                                 \
                                                                                                                    \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                          \
        sleep(2);                                                                                                   \
                                                                                                                    \
        ASSERT_EQ(aclrtMemcpy(output.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);           \
        int32_t flag = 0;                                                                                           \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            if (output[i] != input[i]) {                                                                            \
                flag = 1;                                                                                           \
            }                                                                                                       \
        }                                                                                                           \
        ASSERT_EQ(flag, 0);                                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(PUT_SIZE_NBI)
#undef PUT_SIZE_NBI

#define PUT_SIZE_NBI(BITS, TYPE)                                                                                    \
    void host_test_aclshmem_##BITS##_putmem_nbi(int rank_id, int n_ranks, uint64_t local_mem_size)                  \
    {                                                                                                               \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                               \
        aclrtStream stream;                                                                                         \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                                       \
        ASSERT_NE(stream, nullptr);                                                                                 \
        host_test_##BITS##_putmem_nbi(rank_id);                                                                     \
                                                                                                                    \
        test_finalize(stream, device_id);                                                                           \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(PUT_SIZE_NBI)
#undef PUT_SIZE_NBI

#define PUT_SIZE_NBI(BITS, TYPE)                                                                                    \
    TEST(TestMemHostApi, TestShmemPutSize##BITS##MemNbi)                                                            \
    {                                                                                                               \
        const int process_count = test_gnpu_num;                                                                    \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                           \
        test_mutil_task(                                                                                            \
            [](int rank_id, int n_rank, uint64_t local_memsize) {                                                   \
                host_test_aclshmem_##BITS##_putmem_nbi(rank_id, n_rank, local_memsize);                             \
            },                                                                                                      \
            local_mem_size, process_count);                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(PUT_SIZE_NBI)
#undef PUT_SIZE_NBI


#define GET_SIZE(BITS, TYPE)                                                                                        \
    static void host_test_##BITS##_getmem(uint32_t rank_id)                                                         \
    {                                                                                                               \
        size_t input_size = elem_size * sizeof(TYPE);                                                               \
        std::vector<TYPE> input(elem_size, rank_id);                                                                \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            input[i] = i + 1;                                                                                       \
        }                                                                                                           \
        std::vector<TYPE> output(elem_size, 0);                                                                     \
                                                                                                                    \
        void *dev_ptr;                                                                                              \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                \
        void *ptr = aclshmem_malloc(1024);                                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);            \
                                                                                                                    \
        aclshmem_get##BITS(dev_ptr, ptr, elem_size, rank_id);                                                       \
                                                                                                                    \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                          \
        sleep(2);                                                                                                   \
                                                                                                                    \
        ASSERT_EQ(aclrtMemcpy(output.data(), input_size, dev_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);       \
        int32_t flag = 0;                                                                                           \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            if (output[i] != input[i]) {                                                                            \
                flag = 1;                                                                                           \
            }                                                                                                       \
        }                                                                                                           \
        ASSERT_EQ(flag, 0);                                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(GET_SIZE)
#undef GET_SIZE

#define GET_SIZE(BITS, TYPE)                                                                                        \
    void host_test_aclshmem_##BITS##_getmem(int rank_id, int n_ranks, uint64_t local_mem_size)                      \
    {                                                                                                               \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                               \
        aclrtStream stream;                                                                                         \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                                       \
        ASSERT_NE(stream, nullptr);                                                                                 \
        host_test_##BITS##_getmem(rank_id);                                                                         \
                                                                                                                    \
        test_finalize(stream, device_id);                                                                           \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(GET_SIZE)
#undef GET_SIZE

#define GET_SIZE(BITS, TYPE)                                                                                        \
    TEST(TestMemHostApi, TestShmemGetSize##BITS##Mem)                                                               \
    {                                                                                                               \
        const int process_count = test_gnpu_num;                                                                    \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                           \
        test_mutil_task(                                                                                            \
            [](int rank_id, int n_rank, uint64_t local_memsize) {                                                   \
                host_test_aclshmem_##BITS##_getmem(rank_id, n_rank, local_memsize);                                 \
            },                                                                                                      \
            local_mem_size, process_count);                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(GET_SIZE)
#undef GET_SIZE


#define GET_SIZE_NBI(BITS, TYPE)                                                                                    \
    static void host_test_##BITS##_getmem_nbi(uint32_t rank_id)                                                     \
    {                                                                                                               \
        size_t input_size = elem_size * sizeof(TYPE);                                                               \
        std::vector<TYPE> input(elem_size, rank_id);                                                                \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            input[i] = i + 1;                                                                                       \
        }                                                                                                           \
        std::vector<TYPE> output(elem_size, 0);                                                                     \
                                                                                                                    \
        void *dev_ptr;                                                                                              \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                \
        void *ptr = aclshmem_malloc(1024);                                                                          \
        ASSERT_EQ(aclrtMemcpy(ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);            \
                                                                                                                    \
        aclshmem_get##BITS##_nbi(dev_ptr, ptr, elem_size, rank_id);                                                 \
                                                                                                                    \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                          \
        sleep(2);                                                                                                   \
                                                                                                                    \
        ASSERT_EQ(aclrtMemcpy(output.data(), input_size, dev_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);       \
        int32_t flag = 0;                                                                                           \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            if (output[i] != input[i]) {                                                                            \
                flag = 1;                                                                                           \
            }                                                                                                       \
        }                                                                                                           \
        ASSERT_EQ(flag, 0);                                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(GET_SIZE_NBI)
#undef GET_SIZE_NBI

#define GET_SIZE_NBI(BITS, TYPE)                                                                                    \
    void host_test_aclshmem_##BITS##_getmem_nbi(int rank_id, int n_ranks, uint64_t local_mem_size)                  \
    {                                                                                                               \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                               \
        aclrtStream stream;                                                                                         \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                                       \
        ASSERT_NE(stream, nullptr);                                                                                 \
        host_test_##BITS##_getmem_nbi(rank_id);                                                                     \
                                                                                                                    \
        test_finalize(stream, device_id);                                                                           \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(GET_SIZE_NBI)
#undef GET_SIZE_NBI

#define GET_SIZE_NBI(BITS, TYPE)                                                                                    \
    TEST(TestMemHostApi, TestShmemGetSize##BITS##MemNbi)                                                            \
    {                                                                                                               \
        const int process_count = test_gnpu_num;                                                                    \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                           \
        test_mutil_task(                                                                                            \
            [](int rank_id, int n_rank, uint64_t local_memsize) {                                                   \
                host_test_aclshmem_##BITS##_getmem_nbi(rank_id, n_rank, local_memsize);                             \
            },                                                                                                      \
            local_mem_size, process_count);                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(GET_SIZE_NBI)
#undef GET_SIZE_NBI