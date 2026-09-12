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
#include "shmem.h"
#include "utils/func_type.h"

using namespace std;

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

#define put_SIGNAL(NAME, TYPE)                                                                                  \
    class HostPut##NAME##memSignal {                                                                            \
    public:                                                                                                     \
        inline HostPut##NAME##memSignal()                                                                       \
        {                                                                                                       \
        }                                                                                                       \
        inline void Init(TYPE *gva, TYPE *dev, uint8_t *sig_addr_, int32_t signal_, int64_t rank_, int sig_op_) \
        {                                                                                                       \
            gva_gm = gva;                                                                                       \
            dev_gm = dev;                                                                                       \
            sig_addr = sig_addr_;                                                                               \
                                                                                                                \
            signal = signal_;                                                                                   \
            rank = rank_;                                                                                       \
            sig_op = sig_op_;                                                                                   \
        }                                                                                                       \
        inline void Process()                                                                                   \
        {                                                                                                       \
            aclshmem_##NAME##_put_signal(gva_gm, dev_gm, 16, sig_addr, signal, sig_op, rank);                   \
        }                                                                                                       \
                                                                                                                \
    private:                                                                                                    \
        TYPE *gva_gm;                                                                                           \
        TYPE *dev_gm;                                                                                           \
        uint8_t *sig_addr;                                                                                      \
        int32_t signal;                                                                                         \
        int sig_op;                                                                                             \
        int64_t rank;                                                                                           \
    };
ACLSHMEM_MEM_PUT_GET_FUNC(put_SIGNAL)
#undef put_SIGNAL

#define put_SIGNAL(NAME, TYPE)                                                                              \
    void putmem_##NAME##_signal_test(TYPE *gva, TYPE *dev, uint8_t *sig_addr, int32_t signal, int64_t rank, \
                                     int sig_op)                                                            \
    {                                                                                                       \
        HostPut##NAME##memSignal op;                                                                        \
        op.Init(gva, dev, sig_addr, signal, rank, sig_op);                                                  \
        op.Process();                                                                                       \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(put_SIGNAL)
#undef put_SIGNAL

#define put_SIGNAL(NAME, TYPE)                                                                                      \
    static void host_test_##NAME##_putmem_signal(uint32_t rank_id, int sig_op)                                      \
    {                                                                                                               \
        size_t input_size = 16 * sizeof(TYPE);                                                                      \
        std::vector<TYPE> input(16, rank_id);                                                                       \
        for (int i = 0; i < 16; i++) {                                                                              \
            input[i] = i + 1;                                                                                       \
        }                                                                                                           \
        std::cout << std::endl;                                                                                     \
        std::vector<TYPE> output(32, 0);                                                                            \
        std::vector<int32_t> output_signal(1, 0);                                                                   \
                                                                                                                    \
        void *dev_ptr;                                                                                              \
        void *signal_addr;                                                                                          \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                \
        ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);        \
        ASSERT_EQ(aclrtMalloc(&signal_addr, sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY), 0);                       \
        void *ptr = aclshmem_malloc(1024);                                                                          \
        int32_t signal = 6;                                                                                         \
        putmem_##NAME##_signal_test((TYPE *)ptr, (TYPE *)dev_ptr, (uint8_t *)signal_addr, signal, rank_id, sig_op); \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                          \
        sleep(2);                                                                                                   \
                                                                                                                    \
        ASSERT_EQ(aclrtMemcpy(output.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);           \
        ASSERT_EQ(aclrtMemcpy(output_signal.data(), sizeof(int32_t), signal_addr, sizeof(int32_t),                  \
                              ACL_MEMCPY_DEVICE_TO_HOST), 0);                                                       \
        int32_t flag = 0;                                                                                           \
        for (int i = 0; i < 4; i++) {                                                                               \
            if (output[i] != input[i]) {                                                                            \
                flag = 1;                                                                                           \
            }                                                                                                       \
        }                                                                                                           \
        ASSERT_EQ(flag, 0);                                                                                         \
        ASSERT_EQ(output_signal[0], 6);                                                                             \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(put_SIGNAL)
#undef put_SIGNAL

#define put_SIGNAL(NAME, TYPE)                                                                                  \
    void host_test_aclshmem_##NAME##_mem_signal(int rank_id, int n_ranks, uint64_t local_mem_size, int sig_op)  \
    {                                                                                                           \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                           \
        aclrtStream stream;                                                                                     \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                                   \
        ASSERT_NE(stream, nullptr);                                                                             \
        host_test_##NAME##_putmem_signal(rank_id, sig_op);                                                      \
                                                                                                                \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                            \
        test_finalize(stream, device_id);                                                                       \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(put_SIGNAL)
#undef put_SIGNAL

#define put_SIGNAL_NBI(NAME, TYPE)                                                                              \
    class HostPut##NAME##memSignalNbi {                                                                         \
    public:                                                                                                     \
        inline HostPut##NAME##memSignalNbi()                                                                    \
        {                                                                                                       \
        }                                                                                                       \
        inline void Init(TYPE *gva, TYPE *dev, uint8_t *sig_addr_, int32_t signal_, int64_t rank_, int sig_op_) \
        {                                                                                                       \
            gva_gm = gva;                                                                                       \
            dev_gm = dev;                                                                                       \
            sig_addr = sig_addr_;                                                                               \
                                                                                                                \
            signal = signal_;                                                                                   \
            rank = rank_;                                                                                       \
            sig_op = sig_op_;                                                                                   \
        }                                                                                                       \
        inline void Process()                                                                                   \
        {                                                                                                       \
            aclshmem_##NAME##_put_signal_nbi(gva_gm, dev_gm, 16, sig_addr, signal, sig_op, rank);               \
        }                                                                                                       \
                                                                                                                \
    private:                                                                                                    \
        TYPE *gva_gm;                                                                                           \
        TYPE *dev_gm;                                                                                           \
        uint8_t *sig_addr;                                                                                      \
        int32_t signal;                                                                                         \
        int sig_op;                                                                                             \
        int64_t rank;                                                                                           \
    };
ACLSHMEM_MEM_PUT_GET_FUNC(put_SIGNAL_NBI)
#undef put_SIGNAL_NBI

#define put_SIGNAL_NBI(NAME, TYPE)                                                                              \
    void putmem_signal_##NAME##_test_nbi(TYPE *gva, TYPE *dev, uint8_t *sig_addr, int32_t signal, int64_t rank, \
                                         int sig_op)                                                            \
    {                                                                                                           \
        HostPut##NAME##memSignalNbi op;                                                                         \
        op.Init(gva, dev, sig_addr, signal, rank, sig_op);                                                      \
        op.Process();                                                                                           \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(put_SIGNAL_NBI)
#undef put_SIGNAL_NBI

#define put_SIGNAL_NBI(NAME, TYPE)                                                                             \
    static void host_test_##NAME##_putmem_signal_nbi(uint32_t rank_id, int sig_op)                             \
    {                                                                                                          \
        size_t input_size = 16 * sizeof(TYPE);                                                                 \
        std::vector<TYPE> input(16, rank_id);                                                                  \
        for (int i = 0; i < 16; i++) {                                                                         \
            input[i] = i + 1;                                                                                  \
        }                                                                                                      \
        std::vector<TYPE> output(16, 0);                                                                       \
        std::vector<int32_t> output_signal(1, 0);                                                              \
                                                                                                               \
        void *dev_ptr;                                                                                         \
        void *signal_addr;                                                                                     \
        ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                           \
        ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);   \
        ASSERT_EQ(aclrtMalloc(&signal_addr, sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY), 0);                  \
        void *ptr = aclshmem_malloc(1024);                                                                     \
        int32_t signal = 6;                                                                                    \
        putmem_signal_##NAME##_test_nbi((TYPE *)ptr, (TYPE *)dev_ptr, (uint8_t *)signal_addr, signal, rank_id, \
                                        sig_op);                                                               \
        ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);                                     \
        sleep(2);                                                                                              \
                                                                                                               \
        ASSERT_EQ(aclrtMemcpy(output.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);      \
        ASSERT_EQ(aclrtMemcpy(output_signal.data(), sizeof(int32_t), signal_addr, sizeof(int32_t),             \
                              ACL_MEMCPY_DEVICE_TO_HOST), 0);                                                  \
        int32_t flag = 0;                                                                                      \
        for (int i = 0; i < 16; i++) {                                                                         \
            if (output[i] != input[i]) {                                                                       \
                flag = 1;                                                                                      \
            }                                                                                                  \
        }                                                                                                      \
        ASSERT_EQ(flag, 0);                                                                                    \
        ASSERT_EQ(output_signal[0], 6);                                                                        \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(put_SIGNAL_NBI)
#undef put_SIGNAL_NBI

#define put_SIGNAL_NBI(NAME, TYPE)                                                                                  \
    void host_test_aclshmem_##NAME##_mem_signal_nbi(int rank_id, int n_ranks, uint64_t local_mem_size, int sig_op)  \
    {                                                                                                               \
        int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                                               \
        aclrtStream stream;                                                                                         \
        test_init(rank_id, n_ranks, local_mem_size, &stream);                                                       \
        ASSERT_NE(stream, nullptr);                                                                                 \
        host_test_##NAME##_putmem_signal_nbi(rank_id, sig_op);                                                      \
                                                                                                                    \
        std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                                \
        test_finalize(stream, device_id);                                                                           \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(put_SIGNAL_NBI)
#undef put_SIGNAL_NBI

#define put_SIGNAL_NBI(NAME, TYPE)                                                                          \
    TEST(TestMemHostApi, TestShmemPut##NAME##MemSignal)                                                     \
    {                                                                                                       \
        const int process_count = test_gnpu_num;                                                            \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                   \
        test_mutil_task(                                                                                    \
            [](int rank_id, int n_rank, uint64_t local_memsize) {                                           \
                host_test_aclshmem_##NAME##_mem_signal(rank_id, n_rank, local_memsize, ACLSHMEM_SIGNAL_SET);\
            },                                                                                              \
            local_mem_size, process_count);                                                                 \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(put_SIGNAL_NBI)
#undef put_SIGNAL_NBI

#define put_SIGNAL_NBI(NAME, TYPE)                                                                              \
    TEST(TestMemHostApi, TestShmemPut##NAME##MemSignalNbi)                                                      \
    {                                                                                                           \
        const int process_count = test_gnpu_num;                                                                \
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;                                                       \
        test_mutil_task(                                                                                        \
            [](int rank_id, int n_rank, uint64_t local_memsize) {                                               \
                host_test_aclshmem_##NAME##_mem_signal_nbi(rank_id, n_rank, local_memsize, ACLSHMEM_SIGNAL_SET);\
            },                                                                                                  \
            local_mem_size, process_count);                                                                     \
    }
ACLSHMEM_MEM_PUT_GET_FUNC(put_SIGNAL_NBI)
#undef put_SIGNAL_NBI