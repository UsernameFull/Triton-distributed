/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "multi_instance_kernel.h"
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

#undef inline
#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"
#define inline inline attribute((always_inline))

using namespace AscendC;

using fp16_t = op::fp16_t;
using bf16_t = op::bfloat16;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;

template <typename T>
ACLSHMEM_DEVICE void all_gather_origin(__gm__ T *input, __gm__ T *output, __gm__ T *gva, int64_t max_gva_num,
                                    int elements, int len, int64_t magic)
{
    const int64_t aivNum = GetBlockNum();
    const int64_t aivIndex = GetBlockIdx();

    const int64_t data_offset = aivNum * SYNC_FLAG_INTERVAL;
    const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    __gm__ T *input_gm = (__gm__ T *)input;
    __gm__ T *output_gm = (__gm__ T *)output;
    __gm__ T *gva_data_gm = (__gm__ T *)((__gm__ int32_t *)gva + data_offset);
    __gm__ int32_t *gva_sync_gm = (__gm__ int32_t *)gva;

    // signal_op needed
    __ubuf__ int32_t *flags_ub1[16];
    __ubuf__ int32_t *flags_ub2[16];
    for (int i = 0; i * 8 < 128; i++) {
        flags_ub1[i] = (__ubuf__ int32_t *)(32) + i * 16;
        flags_ub2[i] = (__ubuf__ int32_t *)(544) + i * 16;
    }

    // 0-7 copy data to local symmetric mem, 8-15 copy remote data from symmetric mem.
    int core_group_num = aivNum / 2;
    int core_per_rank = core_group_num / pe_size;
    int len_per_core = len / core_group_num;

    int group_per_num = len_per_core;
    if (aivIndex == core_group_num - 1) {  // Remain Handle
        group_per_num = len - group_per_num * aivIndex;
    }

    // GM to SymmPtr
    if (aivIndex < core_group_num) {
        __ubuf__ T *tmp_buff = reinterpret_cast<__ubuf__ T *>(uint64_t(1024 + 32));
        uint32_t copy_ub_size = UB_DMA_MAX_SIZE;
        uint32_t copy_ub_num = copy_ub_size / sizeof(T);
        uint32_t copy_total_size = group_per_num * sizeof(T);

        int64_t times = 0;
        int64_t flag = 0;
        while (copy_total_size >= copy_ub_size) {
            aclshmemx_mte_put_nbi(gva_data_gm + aivIndex * len_per_core + times * copy_ub_num,
                                  input_gm + aivIndex * len_per_core + times * copy_ub_num, tmp_buff, copy_ub_size,
                                  copy_ub_num, my_rank, EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            times += 1;
            flag = times + magic;
            aclshmemx_signal_op(gva_sync_gm + flag_offset, flag, ACLSHMEM_SIGNAL_SET, my_rank);

            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            copy_total_size -= copy_ub_size;
        }
        if (copy_total_size <= 0) {
            return;
        }
        aclshmemx_mte_put_nbi(gva_data_gm + aivIndex * len_per_core + times * copy_ub_num,
                              input_gm + aivIndex * len_per_core + times * copy_ub_num, tmp_buff, copy_ub_size,
                              copy_total_size / sizeof(T), my_rank, EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        times += 1;
        flag = times + magic;
        aclshmemx_signal_op(gva_sync_gm + flag_offset, flag, ACLSHMEM_SIGNAL_SET, my_rank);
        return;
    }

    // while style
    for (int64_t i = 0; i < core_group_num; i++) {
        *flags_ub1[i] = 0;
        *flags_ub2[i] = 0;
    }

    __ubuf__ T *ping_buff = reinterpret_cast<__ubuf__ T *>(uint64_t(1 * 1024 + 32));
    __ubuf__ T *pong_buff = reinterpret_cast<__ubuf__ T *>(uint64_t(96 * 1024 + 32));
    uint32_t copy_ub_size = UB_DMA_MAX_SIZE / 2;
    uint32_t copy_ub_num = copy_ub_size / sizeof(T);
    int x = (aivIndex - core_group_num) / core_per_rank;

    int pingpongId = 0;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    while (true) {
        for (int group_idx = 0; group_idx < core_group_num; group_idx++) {
            if (*flags_ub1[group_idx] == INT32_MAX) {
                continue;
            }

            int64_t all_data_size = len_per_core * sizeof(T);
            if (group_idx == core_group_num - 1) {
                all_data_size = (len - group_idx * len_per_core) * sizeof(T);
            }

            if (*flags_ub1[group_idx] * UB_DMA_MAX_SIZE >= all_data_size) {
                *flags_ub1[group_idx] = INT32_MAX;
                continue;
            }

            aclshmem_int32_get_nbi(flags_ub2[group_idx], gva_sync_gm + group_idx * SYNC_FLAG_INTERVAL, 1, x);
            AscendC::PipeBarrier<PIPE_ALL>();

            if ((*flags_ub2[group_idx] >> 10) != (magic >> 10)) {
                continue;
            }

            int64_t ready_num = *flags_ub2[group_idx] - magic;
            if (ready_num <= 0 || *flags_ub1[group_idx] >= ready_num) {
                continue;
            }

            int group_recv_offset = x * elements + group_idx * len_per_core;
            int group_send_offset = group_idx * len_per_core;

            int send_offset = *flags_ub1[group_idx] * UB_DMA_MAX_SIZE / sizeof(T);
            int recv_offset = *flags_ub1[group_idx] * UB_DMA_MAX_SIZE / sizeof(T);
            int num_total = (ready_num - *flags_ub1[group_idx]) * UB_DMA_MAX_SIZE / sizeof(T);
            if (ready_num * UB_DMA_MAX_SIZE > all_data_size) {
                num_total = (all_data_size - *flags_ub1[group_idx] * UB_DMA_MAX_SIZE) / sizeof(T);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int i = 0; num_total > 0; i++) {
                AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
                __ubuf__ T *buf = pingpongId == 0 ? ping_buff : pong_buff;

                uint32_t copy_num = num_total > copy_ub_num ? copy_ub_num : num_total;

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
                aclshmemx_mte_get_nbi(output_gm + group_recv_offset + recv_offset,
                                      gva_data_gm + group_send_offset + send_offset, buf, copy_ub_size, copy_num, x,
                                      EVENT_ID);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);

                send_offset += copy_num;
                recv_offset += copy_num;
                num_total -= copy_num;
                pingpongId = 1 - pingpongId;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            *flags_ub1[group_idx] = ready_num;
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        bool finished = true;
        for (int64_t group_idx = 0; group_idx < core_group_num; group_idx++) {
            if (*flags_ub1[group_idx] != INT32_MAX) {
                finished = false;
                break;
            }
        }
        if (finished) {
            break;
        }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// all_gather
template <typename T>
ACLSHMEM_DEVICE void all_gather_big_data(uint64_t fftsAddr, __gm__ T *input, __gm__ T *output, __gm__ T *gva,
                                      int elements, int magic)
{
#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    util_set_ffts_config(fftsAddr);

    const int64_t max_gva_num = GVA_BUFF_MAX_SIZE / sizeof(T);
    int times = (elements + max_gva_num - 1) / max_gva_num;
    int total_num = elements;

    __ubuf__ int64_t *ctrl_ub = (__ubuf__ int64_t *)(0);
    for (int i = 0; i < times; i++) {
        *ctrl_ub = 0;
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t len = total_num > max_gva_num ? max_gva_num : total_num;
        aclshmemx_barrier_all_vec();
        all_gather_origin(input + i * max_gva_num, output + i * max_gva_num, gva, max_gva_num,
                            elements, len, (magic + i) * 1024);

        total_num -= max_gva_num;
        AscendC::PipeBarrier<PIPE_ALL>();
    }
#endif
}

// all_gather
template <typename T>
ACLSHMEM_DEVICE void all_gather_small_data(uint64_t fftsAddr, __gm__ T *input, __gm__ T *output, __gm__ T *gva,
                                        int elements, int magic)
{
#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    const int64_t aivNum = GetBlockNum();
    const int64_t aivIndex = GetBlockIdx();

    const int64_t data_offset = aivNum * SYNC_FLAG_INTERVAL;
    const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    __gm__ T *input_gm = (__gm__ T *)input;
    __gm__ T *output_gm = (__gm__ T *)output;

    __gm__ T *gva_data_gm = (__gm__ T *)((__gm__ int32_t *)gva + data_offset);
    __gm__ int32_t *gva_sync_gm = (__gm__ int32_t *)gva;

    __ubuf__ T *tmp_buff = (__ubuf__ T *)(64);

    // data move parameters
    const uint32_t ub_size = UB_DMA_MAX_SIZE;
    uint32_t input_offset, output_offset, gva_offset, num_per_core;

    // [AllGather Step 1] local input gm -> symmetric mem.
    num_per_core = elements / aivNum;
    input_offset = aivIndex * num_per_core;
    gva_offset = aivIndex * num_per_core;
    if (aivIndex == aivNum - 1) {
        num_per_core = elements - num_per_core * aivIndex;
    }
    aclshmemx_mte_put_nbi(gva_data_gm + gva_offset, input_gm + input_offset, tmp_buff, ub_size, num_per_core, my_rank,
                          EVENT_ID0);

    const int64_t core_per_rank = aivNum / pe_size;
    const int64_t core_rank_idx = aivIndex % core_per_rank;
    const int64_t x = aivIndex / core_per_rank;

    // Sync Ensure Corresponding Tasks Done.
    aclshmem_quiet();
    aclshmemi_barrier_core_soft();

    aclshmemx_signal_op(gva_sync_gm + flag_offset, magic, ACLSHMEM_SIGNAL_SET, my_rank);
    aclshmem_signal_wait_until((__gm__ int32_t *)aclshmem_ptr(gva_sync_gm, x) + flag_offset, ACLSHMEM_CMP_EQ, magic);

    // [AllGather Step 2] symmetric mem -> local output.
    num_per_core = elements / core_per_rank;
    output_offset = x * elements + core_rank_idx * num_per_core;
    gva_offset = core_rank_idx * num_per_core;
    if (core_rank_idx == core_per_rank - 1) {
        num_per_core = elements - num_per_core * core_rank_idx;
    }
    aclshmemx_mte_get_nbi(output_gm + output_offset, gva_data_gm + gva_offset, tmp_buff, ub_size, num_per_core, x,
                          EVENT_ID0);
#endif
}

#define ALLGATHER_FUNC_DEF(type)                                                                                                \
    extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void ShmemAllGather_##type(uint64_t fftsAddr, GM_ADDR input, GM_ADDR output,               \
                                        GM_ADDR gva, int elements, int magic, uint64_t instance_id)                             \
    {                                                                                                                           \
        /* Set multi-instance context. */                                                                                       \
        aclshmemx_instance_ctx_set(instance_id);                                                                                \
                                                                                                                                \
        if (elements * sizeof(type) < 2097152) {                                                                                \
            all_gather_small_data<type>(                                                                                        \
                fftsAddr, (__gm__ type *)input, (__gm__ type *)output, (__gm__ type *)gva, elements, magic);                    \
        } else {                                                                                                                \
            all_gather_big_data<type>(                                                                                          \
                fftsAddr, (__gm__ type *)input, (__gm__ type *)output, (__gm__ type *)gva, elements, magic);                    \
        }                                                                                                                       \
    }

#define TYPE_FUNC(fun) \
    fun(int);          \
    fun(int32_t);      \
    fun(float16_t);    \
    fun(bfloat16_t)

TYPE_FUNC(ALLGATHER_FUNC_DEF);

template <class T>
void group_allgather(uint32_t block_dim, void *stream, uint64_t fftsAddr,
    uint8_t *input, uint8_t *output, uint8_t *gva, int elements, int magic, uint64_t instance_id)
{
    if (std::is_same<T, int>::value) {
        ShmemAllGather_int<<<block_dim, nullptr, stream>>>(fftsAddr, input, output, gva, elements, magic, instance_id);
    } else if (std::is_same<T, int32_t>::value) {
        ShmemAllGather_int32_t<<<block_dim, nullptr, stream>>>(fftsAddr, input, output, gva, elements, magic, instance_id);
    } else if (std::is_same<T, fp16_t>::value) {
        ShmemAllGather_float16_t<<<block_dim, nullptr, stream>>>(fftsAddr, input, output, gva, elements, magic, instance_id);
    } else if (std::is_same<T, bf16_t>::value) {
        ShmemAllGather_bfloat16_t<<<block_dim, nullptr, stream>>>(fftsAddr, input, output, gva, elements, magic, instance_id);
    }
}

template void group_allgather<int>
    (uint32_t block_dim, void *stream, uint64_t fftsAddr, uint8_t *input, uint8_t *output, uint8_t *gva, int elements, int magic, uint64_t instance_id);
template void group_allgather<fp16_t>
    (uint32_t block_dim, void *stream, uint64_t fftsAddr, uint8_t *input, uint8_t *output, uint8_t *gva, int elements, int magic, uint64_t instance_id);
template void group_allgather<bf16_t>
    (uint32_t block_dim, void *stream, uint64_t fftsAddr, uint8_t *input, uint8_t *output, uint8_t *gva, int elements, int magic, uint64_t instance_id);