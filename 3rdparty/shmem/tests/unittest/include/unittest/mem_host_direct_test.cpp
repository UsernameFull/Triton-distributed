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
#include "shmem.h"
#include "shmemi_host_common.h"
#include "utils/func_type.h"

using namespace std;

constexpr int Int = 1;

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

extern void aclshmem_barrier_all();

#define put_TEST(NAME, TYPE)                                                                                       \
    class HostPut##NAME##Test {                                                                                    \
    public:                                                                                                        \
        inline HostPut##NAME##Test()                                                                               \
        {                                                                                                          \
        }                                                                                                          \
        inline void Init(uint8_t *gva, uint8_t *dev, int64_t rank_, int64_t rank_size_)                            \
        {                                                                                                          \
            gva_gm = reinterpret_cast<TYPE *>(gva);                                                                \
            dev_gm = reinterpret_cast<TYPE *>(dev);                                                                \
                                                                                                                   \
            rank = rank_;                                                                                          \
            rank_size = rank_size_;                                                                                \
        }                                                                                                          \
        inline void Process(bool is_nbi = false)                                                                   \
        {                                                                                                          \
            if (is_nbi) {                                                                                          \
                aclshmem_##NAME##_put_nbi(gva_gm, dev_gm, rank_size * 16, rank);                                   \
            } else {                                                                                               \
                aclshmem_##NAME##_put(gva_gm, dev_gm, rank_size * 16, rank);                                       \
            }                                                                                                      \
        }                                                                                                          \
                                                                                                                   \
    private:                                                                                                       \
        TYPE *gva_gm;                                                                                              \
        TYPE *dev_gm;                                                                                              \
                                                                                                                   \
        int64_t rank;                                                                                              \
        int64_t rank_size;                                                                                         \
    };                                                                                                             \
    void host_test_##NAME##_put(uint8_t *gva, uint8_t *dev, int64_t rank_, int64_t rank_size_, bool is_nbi = true) \
    {                                                                                                              \
        HostPut##NAME##Test op;                                                                                    \
        op.Init(gva, dev, rank_, rank_size_);                                                                      \
        op.Process(is_nbi);                                                                                        \
    }

ACLSHMEM_MEM_PUT_GET_FUNC(put_TEST)

#define get_TEST(NAME, TYPE)                                                                                       \
    class HostGet##NAME##Test {                                                                                    \
    public:                                                                                                        \
        inline HostGet##NAME##Test()                                                                               \
        {                                                                                                          \
        }                                                                                                          \
        inline void Init(uint8_t *gva, uint8_t *dev, int64_t rank_, int64_t rank_size_)                            \
        {                                                                                                          \
            gva_gm = reinterpret_cast<TYPE *>(gva);                                                                \
            dev_gm = reinterpret_cast<TYPE *>(dev);                                                                \
                                                                                                                   \
            rank = rank_;                                                                                          \
            rank_size = rank_size_;                                                                                \
        }                                                                                                          \
        inline void Process(bool is_nbi = false)                                                                   \
        {                                                                                                          \
            if (is_nbi) {                                                                                          \
                for (int i = 0; i < rank_size; i++) {                                                              \
                    aclshmem_##NAME##_get_nbi(dev_gm + 16 * i, gva_gm, 16, i % rank_size);                         \
                }                                                                                                  \
            } else {                                                                                               \
                for (int i = 0; i < rank_size; i++) {                                                              \
                    aclshmem_##NAME##_get(dev_gm + 16 * i, gva_gm, 16, i % rank_size);                             \
                }                                                                                                  \
            }                                                                                                      \
        }                                                                                                          \
                                                                                                                   \
    private:                                                                                                       \
        TYPE *gva_gm;                                                                                              \
        TYPE *dev_gm;                                                                                              \
                                                                                                                   \
        int64_t rank;                                                                                              \
        int64_t rank_size;                                                                                         \
    };                                                                                                             \
                                                                                                                   \
    void host_test_##NAME##_get(uint8_t *gva, uint8_t *dev, int64_t rank_, int64_t rank_size_, bool is_nbi = true) \
    {                                                                                                              \
        HostGet##NAME##Test op;                                                                                    \
        op.Init(gva, dev, rank_, rank_size_);                                                                      \
        op.Process(is_nbi);                                                                                        \
    }

ACLSHMEM_MEM_PUT_GET_FUNC(get_TEST)

#define PUT_GET_TEST(NAME, TYPE)                                                                                   \
    static void host_test_##NAME##_put_get(uint8_t *gva, uint32_t rank_id, uint32_t rank_size, bool is_nbi = true) \
    {                                                                                                              \
        int total_size = 16 * static_cast<int>(rank_size);                                                         \
        size_t input_size = total_size * sizeof(TYPE);                                                             \
        std::vector<TYPE> input(total_size, 0);                                                                    \
        for (int i = 0; i < 16; i++) {                                                                             \
            input[i] = (rank_id + 10);                                                                             \
        }                                                                                                          \
                                                                                                                   \
        void *dev_ptr;                                                                                             \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                               \
                                                                                                                   \
        ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);       \
                                                                                                                   \
        void *ptr = aclshmem_malloc(1024);                                                                         \
        host_test_##NAME##_put((uint8_t *)ptr, (uint8_t *)dev_ptr, rank_id, rank_size, is_nbi);                    \
                                                                                                                   \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                         \
        sleep(2);                                                                                                  \
                                                                                                                   \
        ASSERT_EQ(aclrtMemcpy(input.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);           \
                                                                                                                   \
        string p_name = "[Process " + to_string(rank_id) + "] ";                                                   \
        std::cout << p_name;                                                                                       \
        for (int i = 0; i < total_size; i++) {                                                                     \
            std::cout << static_cast<int>(input[i]) << " ";                                                        \
        }                                                                                                          \
        std::cout << std::endl;                                                                                    \
        host_test_##NAME##_get((uint8_t *)ptr, (uint8_t *)dev_ptr, rank_id, rank_size, is_nbi);                    \
                                                                                                                   \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                         \
        sleep(2);                                                                                                  \
                                                                                                                   \
        ASSERT_EQ(aclrtMemcpy(input.data(), input_size, dev_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);       \
                                                                                                                   \
        std::cout << p_name;                                                                                       \
        for (int i = 0; i < total_size; i++) {                                                                     \
            std::cout << static_cast<int>(input[i]) << " ";                                                        \
        }                                                                                                          \
        std::cout << std::endl;                                                                                    \
        int32_t flag = 0;                                                                                          \
        for (int i = 0; i < total_size; i++) {                                                                     \
            int stage = i / 16;                                                                                    \
            if (input[i] != (stage + 10)) {                                                                        \
                flag = 1;                                                                                          \
            }                                                                                                      \
        }                                                                                                          \
        ASSERT_EQ(flag, 0);                                                                                        \
    }

ACLSHMEM_MEM_PUT_GET_FUNC(PUT_GET_TEST)

#define TEST_MEM(NAME, TYPE)                                                                             \
    void host_test_##NAME##_aclshmem_mem(int rank_id, int n_ranks, uint64_t local_mem_size, bool is_nbi) \
    {                                                                                                    \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                    \
        aclrtStream stream;                                                                              \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                            \
        ASSERT_NE(stream, nullptr);                                                                      \
        host_test_##NAME##_put_get((uint8_t *)g_state.heap_base, rank_id, n_ranks, is_nbi);              \
                                                                                                         \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                     \
        test_finalize(stream, device_id);                                                                \
    }

ACLSHMEM_MEM_PUT_GET_FUNC(TEST_MEM)

#define TEST_API(NAME, TYPE)                                                              \
    TEST(TestMemHostApi, TestShmem##NAME##Mem)                                            \
    {                                                                                     \
        const int process_count = test_gnpu_num;                                          \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                 \
        test_mutil_task(                                                                  \
            [this](int rank_id, int n_ranks, uint64_t local_mem_size) {                   \
                host_test_##NAME##_aclshmem_mem(rank_id, n_ranks, local_mem_size, false); \
            },                                                                            \
            local_mem_size, process_count);                                               \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(TEST_API)
#undef TEST_API

#define TEST_API(NAME, TYPE)                                                             \
    TEST(TestMemHostApi, TestShmemMem##NAME##Nbi)                                        \
    {                                                                                    \
        const int process_count = test_gnpu_num;                                         \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                \
        test_mutil_task(                                                                 \
            [this](int rank_id, int n_ranks, uint64_t local_mem_size) {                  \
                host_test_##NAME##_aclshmem_mem(rank_id, n_ranks, local_mem_size, true); \
            },                                                                           \
            local_mem_size, process_count);                                              \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(TEST_API)
