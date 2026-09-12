/**
?* Copyright (c) 2026 Huawei Technologies Co., Ltd.
?* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
?* CANN Open Software License Agreement Version 2.0 (the "License").
?* Please refer to the License for details. You may not use this file except in compliance with the License.
?* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
?* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
?* See LICENSE in the root of the software repository for the full text of the License.
?*/
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "../utils/func_type.h"

using namespace std;

static constexpr int elem_size = 16 * 1024;

#define IPUT_MEM(NAME, TYPE)                                                                                        \
    static void host_test_aclshmem_##NAME##_iput(uint32_t rank_id)                                                  \
    {                                                                                                               \
        size_t input_size = elem_size * sizeof(TYPE);                                                               \
        std::vector<TYPE> input(elem_size, rank_id);                                                                \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            input[i] = i % 256;                                                                                     \
        }                                                                                                           \
        std::vector<TYPE> output(elem_size, 0);                                                                     \
        std::vector<TYPE> expect(elem_size, 0);                                                                     \
        void *dev_ptr;                                                                                              \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                \
        ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);        \
        TYPE *ptr = static_cast<TYPE *>(aclshmem_malloc(input_size));                                               \
                                                                                                                    \
        ptrdiff_t dst = 3;                                                                                          \
        ptrdiff_t sst = 4;                                                                                          \
        aclshmem_##NAME##_iput(ptr, static_cast<TYPE *>(dev_ptr), dst, sst, elem_size / sst, rank_id);              \
                                                                                                                    \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                          \
        ASSERT_EQ(aclrtMemcpy(output.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);           \
                                                                                                                    \
        for (int i = 0; i < elem_size / sst; i++) {                                                                 \
            expect[i * dst] = input[i * sst];                                                                       \
        }                                                                                                           \
                                                                                                                    \
        int32_t flag = 0;                                                                                           \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            if (output[i] != expect[i]) {                                                                           \
                flag = 1;                                                                                           \
            }                                                                                                       \
        }                                                                                                           \
                                                                                                                    \
        ASSERT_EQ(flag, 0);                                                                                         \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(IPUT_MEM)
#undef IPUT_MEM

#define IPUT_MEM(NAME, TYPE)                                                                                        \
    void host_test_##NAME##_iput(int rank_id, int n_ranks, uint64_t local_mem_size)                                 \
    {                                                                                                               \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                               \
        aclrtStream stream;                                                                                         \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                                       \
        ASSERT_NE(stream, nullptr);                                                                                 \
        host_test_aclshmem_##NAME##_iput(rank_id);                                                                  \
                                                                                                                    \
        test_finalize(stream, device_id);                                                                           \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(IPUT_MEM)
#undef IPUT_MEM

#define IPUT_MEM(NAME, TYPE)                                                                                        \
    TEST(TestMemHostApi, TestShmemIPut##NAME##Mem)                                                                  \
    {                                                                                                               \
        const int process_count = test_gnpu_num;                                                                    \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                           \
        test_mutil_task(                                                                                            \
            [](int rank_id, int n_rank, uint64_t local_memsize) {                                                   \
                host_test_##NAME##_iput(rank_id, n_rank, local_memsize);                                            \
            },                                                                                                      \
            local_mem_size, process_count);                                                                         \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(IPUT_MEM)
#undef IPUT_MEM


#define IGET_MEM(NAME, TYPE)                                                                                        \
    static void host_test_aclshmem_##NAME##_iget(uint32_t rank_id)                                                  \
    {                                                                                                               \
        size_t input_size = elem_size * sizeof(TYPE);                                                               \
        std::vector<TYPE> input(elem_size, rank_id);                                                                \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            input[i] = i % 256;                                                                                     \
        }                                                                                                           \
        std::vector<TYPE> output(elem_size, 0);                                                                     \
        std::vector<TYPE> expect(elem_size, 0);                                                                     \
        void *dev_ptr;                                                                                              \
        TYPE *ptr = static_cast<TYPE *>(aclshmem_malloc(input_size));                                               \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                \
        ASSERT_EQ(aclrtMemcpy(ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);            \
                                                                                                                    \
        ptrdiff_t dst = 3;                                                                                          \
        ptrdiff_t sst = 4;                                                                                          \
        TYPE *dest_ptr = static_cast<TYPE *>(dev_ptr);                                                              \
        aclshmem_##NAME##_iget(dest_ptr, ptr, dst, sst, elem_size / sst, rank_id);                                  \
                                                                                                                    \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                          \
        ASSERT_EQ(aclrtMemcpy(output.data(), input_size, dest_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);      \
                                                                                                                    \
        for (int i = 0; i < elem_size / sst; i++) {                                                                 \
            expect[i * dst] = input[i * sst];                                                                       \
        }                                                                                                           \
                                                                                                                    \
        int32_t flag = 0;                                                                                           \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            if (output[i] != expect[i]) {                                                                           \
                flag = 1;                                                                                           \
            }                                                                                                       \
        }                                                                                                           \
                                                                                                                    \
        ASSERT_EQ(flag, 0);                                                                                         \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(IGET_MEM)
#undef IGET_MEM

#define IGET_MEM(NAME, TYPE)                                                                                        \
    void host_test_##NAME##_iget(int rank_id, int n_ranks, uint64_t local_mem_size)                                 \
    {                                                                                                               \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                               \
        aclrtStream stream;                                                                                         \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                                       \
        ASSERT_NE(stream, nullptr);                                                                                 \
        host_test_aclshmem_##NAME##_iget(rank_id);                                                                  \
                                                                                                                    \
        test_finalize(stream, device_id);                                                                           \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(IGET_MEM)
#undef IGET_MEM

#define IGET_MEM(NAME, TYPE)                                                                                        \
    TEST(TestMemHostApi, TestShmemIGet##NAME##Mem)                                                                  \
    {                                                                                                               \
        const int process_count = test_gnpu_num;                                                                    \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                           \
        test_mutil_task(                                                                                            \
            [](int rank_id, int n_rank, uint64_t local_memsize) {                                                   \
                host_test_##NAME##_iget(rank_id, n_rank, local_memsize);                                            \
            },                                                                                                      \
            local_mem_size, process_count);                                                                         \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(IGET_MEM)
#undef IGET_MEM


#define IPUT_SIZE(BITS, TYPE)                                                                                       \
    static void host_test_aclshmem_iput##BITS(uint32_t rank_id)                                                     \
    {                                                                                                               \
        size_t input_size = elem_size * sizeof(TYPE);                                                               \
        std::vector<TYPE> input(elem_size, rank_id);                                                                \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            input[i] = i % 256;                                                                                     \
        }                                                                                                           \
        std::vector<TYPE> output(elem_size, 0);                                                                     \
        std::vector<TYPE> expect(elem_size, 0);                                                                     \
        void *dev_ptr;                                                                                              \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                \
        ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);        \
        TYPE *ptr = static_cast<TYPE *>(aclshmem_malloc(input_size));                                               \
                                                                                                                    \
        ptrdiff_t dst = 3;                                                                                          \
        ptrdiff_t sst = 4;                                                                                          \
        aclshmem_iput##BITS(ptr, static_cast<TYPE *>(dev_ptr), dst, sst, elem_size / sst, rank_id);                 \
                                                                                                                    \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                          \
        ASSERT_EQ(aclrtMemcpy(output.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);           \
                                                                                                                    \
        for (int i = 0; i < elem_size / sst; i++) {                                                                 \
            expect[i * dst] = input[i * sst];                                                                       \
        }                                                                                                           \
                                                                                                                    \
        int32_t flag = 0;                                                                                           \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            if (output[i] != expect[i]) {                                                                           \
                flag = 1;                                                                                           \
            }                                                                                                       \
        }                                                                                                           \
                                                                                                                    \
        ASSERT_EQ(flag, 0);                                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(IPUT_SIZE)
#undef IPUT_SIZE

#define IPUT_SIZE(BITS, TYPE)                                                                                       \
    void host_test_##BITS##_iput(int rank_id, int n_ranks, uint64_t local_mem_size)                                 \
    {                                                                                                               \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                               \
        aclrtStream stream;                                                                                         \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                                       \
        ASSERT_NE(stream, nullptr);                                                                                 \
        host_test_aclshmem_iput##BITS(rank_id);                                                                     \
                                                                                                                    \
        test_finalize(stream, device_id);                                                                           \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(IPUT_SIZE)
#undef IPUT_SIZE

#define IPUT_SIZE(BITS, TYPE)                                                                                       \
    TEST(TestMemHostApi, TestShmemIPutSize##BITS##Mem)                                                              \
    {                                                                                                               \
        const int process_count = test_gnpu_num;                                                                    \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                           \
        test_mutil_task(                                                                                            \
            [](int rank_id, int n_rank, uint64_t local_memsize) {                                                   \
                host_test_##BITS##_iput(rank_id, n_rank, local_memsize);                                            \
            },                                                                                                      \
            local_mem_size, process_count);                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(IPUT_SIZE)
#undef IPUT_SIZE


#define IGET_SIZE(BITS, TYPE)                                                                                       \
    static void host_test_aclshmem_iget##BITS(uint32_t rank_id)                                                     \
    {                                                                                                               \
        size_t input_size = elem_size * sizeof(TYPE);                                                               \
        std::vector<TYPE> input(elem_size, rank_id);                                                                \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            input[i] = i % 256;                                                                                     \
        }                                                                                                           \
        std::vector<TYPE> output(elem_size, 0);                                                                     \
        std::vector<TYPE> expect(elem_size, 0);                                                                     \
        void *dev_ptr;                                                                                              \
        TYPE *ptr = static_cast<TYPE *>(aclshmem_malloc(input_size));                                               \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                \
        ASSERT_EQ(aclrtMemcpy(ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);            \
                                                                                                                    \
        ptrdiff_t dst = 3;                                                                                          \
        ptrdiff_t sst = 4;                                                                                          \
        TYPE *dest_ptr = static_cast<TYPE *>(dev_ptr);                                                              \
        aclshmem_iget##BITS(dest_ptr, ptr, dst, sst, elem_size / sst, rank_id);                                     \
                                                                                                                    \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                          \
        ASSERT_EQ(aclrtMemcpy(output.data(), input_size, dest_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);      \
                                                                                                                    \
        for (int i = 0; i < elem_size / sst; i++) {                                                                 \
            expect[i * dst] = input[i * sst];                                                                       \
        }                                                                                                           \
                                                                                                                    \
        int32_t flag = 0;                                                                                           \
        for (int i = 0; i < elem_size; i++) {                                                                       \
            if (output[i] != expect[i]) {                                                                           \
                flag = 1;                                                                                           \
            }                                                                                                       \
        }                                                                                                           \
                                                                                                                    \
        ASSERT_EQ(flag, 0);                                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(IGET_SIZE)
#undef IGET_SIZE

#define IGET_SIZE(BITS, TYPE)                                                                                       \
    void host_test_##BITS##_iget(int rank_id, int n_ranks, uint64_t local_mem_size)                                 \
    {                                                                                                               \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                               \
        aclrtStream stream;                                                                                         \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                                       \
        ASSERT_NE(stream, nullptr);                                                                                 \
        host_test_aclshmem_iget##BITS(rank_id);                                                                     \
                                                                                                                    \
        test_finalize(stream, device_id);                                                                           \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(IGET_SIZE)
#undef IGET_SIZE

#define IGET_SIZE(BITS, TYPE)                                                                                       \
    TEST(TestMemHostApi, TestShmemIGetSize##BITS##Mem)                                                              \
    {                                                                                                               \
        const int process_count = test_gnpu_num;                                                                    \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                           \
        test_mutil_task(                                                                                            \
            [](int rank_id, int n_rank, uint64_t local_memsize) {                                                   \
                host_test_##BITS##_iget(rank_id, n_rank, local_memsize);                                            \
            },                                                                                                      \
            local_mem_size, process_count);                                                                         \
    }
ACLSHMEM_PUT_GET_SIZE_FUNC(IGET_SIZE)
#undef IGET_SIZE