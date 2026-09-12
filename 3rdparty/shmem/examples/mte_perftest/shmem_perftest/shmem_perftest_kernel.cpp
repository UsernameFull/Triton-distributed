/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _MTEPERF_KERNEL_SHMEM_PERFTEST_
#define _MTEPERF_KERNEL_SHMEM_PERFTEST_

#include <cstring>
#include "kernel_operator.h"
#include "shmem.h"
#include "utils/prof/shmemi_prof.h"
#include "perftest_common_types.h"

template <typename T>
__aicore__ inline void mte_perf_test_put_impl(GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id, perftest::mte_mode_t test_mode, int ub_size_kb, int64_t prof_pe_val, int loop_count)
{
    int64_t pe = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();
    int64_t prof_pe = prof_pe_val;

    __gm__ T *dst_gm = (__gm__ T *)dst_gva;
    __gm__ T *src_gm = (__gm__ T *)src_gva;

    bool is_unilateral = (test_mode == perftest::TEST_MODE_MTE_PUT || test_mode == perftest::TEST_MODE_MTE_GET);
    if (is_unilateral && pe != prof_pe) {
        aclshmemx_barrier_all_vec();
        return;
    }

    int32_t block_elements = (int32_t)(elements / AscendC::GetBlockNum());
    int32_t current_block_index = AscendC::GetBlockIdx();

    int warmup = 100;
    int loop_test = loop_count;
    int32_t offset;
    int peer_pe = (shmem_my_pe() + 1) % shmem_n_pes();
    if (block_elements * sizeof(T) < 512) {
        offset = 512 / sizeof(T) * current_block_index;
    } else {
        offset = block_elements * current_block_index;
    }
    int ub_size_bytes = ub_size_kb * 1024;
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int i = 0; i < (warmup + loop_test); ++i) {
        if (i >= warmup) {
            SHMEMI_PROF_START(frame_id);
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);

        aclshmemx_mte_put_nbi(dst_gm + offset, src_gm + offset, reinterpret_cast<__ubuf__ T *>(0), ub_size_bytes, block_elements, peer_pe, 0);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);

        if (i >= warmup) {
            SHMEMI_PROF_END(frame_id);
        }
    }
    aclshmemx_barrier_all_vec();
}

template <typename T>
__aicore__ inline void mte_perf_test_get_impl(GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id, perftest::mte_mode_t test_mode, int ub_size_kb, int64_t prof_pe_val, int loop_count)
{
    int64_t pe = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();
    int64_t prof_pe = prof_pe_val;

    __gm__ T *dst_gm = (__gm__ T *)dst_gva;
    __gm__ T *src_gm = (__gm__ T *)src_gva;

    bool is_unilateral = (test_mode == perftest::TEST_MODE_MTE_PUT || test_mode == perftest::TEST_MODE_MTE_GET);
    if (is_unilateral && pe != prof_pe) {
        aclshmemx_barrier_all_vec();
        return;
    }

    int32_t block_elements = (int32_t)(elements / AscendC::GetBlockNum());
    int32_t current_block_index = AscendC::GetBlockIdx();

    int warmup = 100;
    int loop_test = loop_count;
    int32_t offset;
    int peer_pe = (shmem_my_pe() + 1) % shmem_n_pes();
    if (block_elements * sizeof(T) < 512) {
        offset = 512 / sizeof(T) * current_block_index;
    } else {
        offset = block_elements * current_block_index;
    }
    int ub_size_bytes = ub_size_kb * 1024;
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int i = 0; i < (warmup + loop_test); ++i) {
        if (i >= warmup) {
            SHMEMI_PROF_START(frame_id);
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);

        aclshmemx_mte_get_nbi(src_gm + offset, dst_gm + offset, reinterpret_cast<__ubuf__ T *>(0), ub_size_bytes, block_elements, peer_pe, 0);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);

        if (i >= warmup) {
            SHMEMI_PROF_END(frame_id);
        }
    }
    aclshmemx_barrier_all_vec();
}

#define DEFINE_MTE_PERF_KERNEL_FOR_TYPE(type_name, cpp_type) \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_perf_test_##type_name##_put(GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id, perftest::mte_mode_t test_mode, int ub_size_kb, int64_t prof_pe_val, int loop_count) \
{ \
    mte_perf_test_put_impl<cpp_type>(dst_gva, src_gva, elements, frame_id, test_mode, ub_size_kb, prof_pe_val, loop_count); \
} \
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_perf_test_##type_name##_get(GM_ADDR dst_gva, GM_ADDR src_gva, int elements, int32_t frame_id, perftest::mte_mode_t test_mode, int ub_size_kb, int64_t prof_pe_val, int loop_count) \
{ \
    mte_perf_test_get_impl<cpp_type>(dst_gva, src_gva, elements, frame_id, test_mode, ub_size_kb, prof_pe_val, loop_count); \
}

DEFINE_MTE_PERF_KERNEL_FOR_TYPE(float, float)
DEFINE_MTE_PERF_KERNEL_FOR_TYPE(int8, int8_t)
DEFINE_MTE_PERF_KERNEL_FOR_TYPE(int16, int16_t)
DEFINE_MTE_PERF_KERNEL_FOR_TYPE(int32, int32_t)
DEFINE_MTE_PERF_KERNEL_FOR_TYPE(int64, int64_t)
DEFINE_MTE_PERF_KERNEL_FOR_TYPE(uint8, uint8_t)
DEFINE_MTE_PERF_KERNEL_FOR_TYPE(uint16, uint16_t)
DEFINE_MTE_PERF_KERNEL_FOR_TYPE(uint32, uint32_t)
DEFINE_MTE_PERF_KERNEL_FOR_TYPE(uint64, uint64_t)
DEFINE_MTE_PERF_KERNEL_FOR_TYPE(char, char)

#define DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(type_name, cpp_type, test_mode) \
    mte_perf_test_##type_name##_##test_mode<<<block_dim, nullptr, stream>>>(dst_gva, src_gva, elements, frame_id, t_mode, ub_size_kb, prof_pe_val, loop_count)

#define DISPATCH_MTE_PERF_FOR_ALL_TYPES(test_mode) \
    switch (d_type) { \
        case perftest::DATA_TYPE_FLOAT: \
            DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(float, float, test_mode); \
            break; \
        case perftest::DATA_TYPE_INT8: \
            DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(int8, int8_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_INT16: \
            DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(int16, int16_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_INT32: \
            DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(int32, int32_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_INT64: \
            DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(int64, int64_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_UINT8: \
            DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(uint8, uint8_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_UINT16: \
            DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(uint16, uint16_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_UINT32: \
            DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(uint32, uint32_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_UINT64: \
            DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(uint64, uint64_t, test_mode); \
            break; \
        case perftest::DATA_TYPE_CHAR: \
            DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(char, char, test_mode); \
            break; \
        default: \
            DISPATCH_MTE_PERF_FOR_TYPE_AND_TEST(float, float, test_mode); \
            break; \
    }

extern "C" void launch_mte_perf_kernel(uint32_t block_dim, void *stream, uint8_t *dst_gva, uint8_t *src_gva, int elements, int32_t frame_id, int test_mode, int data_type, int ub_size_kb, int64_t prof_pe_val, int loop_count)
{
    perftest::mte_mode_t t_mode = static_cast<perftest::mte_mode_t>(test_mode);
    perftest::perf_data_type_t d_type = static_cast<perftest::perf_data_type_t>(data_type);

    switch (t_mode) {
        case perftest::TEST_MODE_MTE_PUT:
        case perftest::TEST_MODE_BI_PUT:
            DISPATCH_MTE_PERF_FOR_ALL_TYPES(put);
            break;
        case perftest::TEST_MODE_MTE_GET:
        case perftest::TEST_MODE_BI_GET:
            DISPATCH_MTE_PERF_FOR_ALL_TYPES(get);
            break;
        default:
            break;
    }
}

#endif  // _MTEPERF_KERNEL_SHMEM_PERFTEST_
