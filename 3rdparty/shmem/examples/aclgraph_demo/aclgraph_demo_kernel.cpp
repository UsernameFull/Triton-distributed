/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclgraph_demo_kernel.h"
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

#include "utils/prof/shmemi_prof.h"
#include "add_custom.h"

using namespace AscendC;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;

template <typename T>
ACLSHMEM_DEVICE void all_gather_origin(__gm__ T *input, __gm__ T *output, __gm__ T *gva, int64_t max_gva_num,
                                       int elements, int len, int64_t magic)
{
    const int64_t aiv_num = GetBlockNum();
    const int64_t aiv_index = GetBlockIdx();

    const int64_t data_offset = aiv_num * SYNC_FLAG_INTERVAL;
    const int64_t flag_offset = aiv_index * SYNC_FLAG_INTERVAL;

    int32_t my_rank = aclshmem_my_pe();
    int32_t pe_size = aclshmem_n_pes();

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
    int core_group_num = aiv_num / 2;
    int core_per_rank = core_group_num / pe_size;
    int len_per_core = len / core_group_num;

    int group_per_num = len_per_core;
    if (aiv_index == core_group_num - 1) {  // Remain Handle
        group_per_num = len - group_per_num * aiv_index;
    }

    // GM to SymmPtr
    if (aiv_index < core_group_num) {
        __ubuf__ T *tmp_buff = reinterpret_cast<__ubuf__ T *>(uint64_t(1024 + 32));
        uint32_t copy_ub_size = UB_DMA_MAX_SIZE;
        uint32_t copy_ub_num = copy_ub_size / sizeof(T);
        uint32_t copy_total_size = group_per_num * sizeof(T);

        int64_t times = 0;
        int64_t flag = 0;
        while (copy_total_size >= copy_ub_size) {
            SHMEMI_PROF_START(0);
            aclshmemx_mte_put_nbi(gva_data_gm + aiv_index * len_per_core + times * copy_ub_num,
                                  input_gm + aiv_index * len_per_core + times * copy_ub_num, tmp_buff, copy_ub_size,
                                  copy_ub_num, my_rank, EVENT_ID0);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            times += 1;
            flag = times + magic;
            aclshmemx_signal_op(gva_sync_gm + flag_offset, flag, ACLSHMEM_SIGNAL_SET, my_rank);
            SHMEMI_PROF_END(0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);

            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

            copy_total_size -= copy_ub_size;
        }
        if (copy_total_size <= 0) {
            return;
        }
        aclshmemx_mte_put_nbi(gva_data_gm + aiv_index * len_per_core + times * copy_ub_num,
                              input_gm + aiv_index * len_per_core + times * copy_ub_num, tmp_buff, copy_ub_size,
                              copy_total_size / sizeof(T), my_rank, EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
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
    int x = (aiv_index - core_group_num) / core_per_rank;

    int pingpong_id = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
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
            PipeBarrier<PIPE_ALL>();

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
            PipeBarrier<PIPE_ALL>();
            for (int i = 0; num_total > 0; i++) {
                TEventID event_id = pingpong_id == 0 ? EVENT_ID0 : EVENT_ID1;
                __ubuf__ T *buf = pingpong_id == 0 ? ping_buff : pong_buff;

                uint32_t copy_num = num_total > copy_ub_num ? copy_ub_num : num_total;

                WaitFlag<HardEvent::MTE3_MTE2>(event_id);
                aclshmemx_mte_get_nbi(output_gm + group_recv_offset + recv_offset,
                                      gva_data_gm + group_send_offset + send_offset, buf, copy_ub_size, copy_num, x,
                                      event_id);
                SetFlag<HardEvent::MTE3_MTE2>(event_id);

                send_offset += copy_num;
                recv_offset += copy_num;
                num_total -= copy_num;
                pingpong_id = 1 - pingpong_id;
            }
            PipeBarrier<PIPE_ALL>();
            *flags_ub1[group_idx] = ready_num;
            PipeBarrier<PIPE_ALL>();
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
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// all_gather
template <typename T>
ACLSHMEM_DEVICE void all_gather_big_data(uint64_t ffts_addr, __gm__ T *input, __gm__ T *output, __gm__ T *gva, int elements,
                                      __gm__ int *magic)
{
    util_set_ffts_config(ffts_addr);

    int magic_data = *magic;

    const int64_t max_gva_num = GVA_BUFF_MAX_SIZE / sizeof(T);
    int times = (elements + max_gva_num - 1) / max_gva_num;
    int total_num = elements;

    __ubuf__ int64_t *ctrl_ub = (__ubuf__ int64_t *)(0);
    for (int i = 0; i < times; i++) {
        *ctrl_ub = 0;
        PipeBarrier<PIPE_ALL>();
        int32_t len = total_num > max_gva_num ? max_gva_num : total_num;
        aclshmemx_barrier_all_vec();
        all_gather_origin(input + i * max_gva_num, output + i * max_gva_num, gva, max_gva_num, elements, len,
                          (magic_data + i) * 1024);
        total_num -= max_gva_num;
        PipeBarrier<PIPE_ALL>();
    }
}

// all_gather
template <typename T>
ACLSHMEM_DEVICE void all_gather_small_data(uint64_t ffts_addr, __gm__ T *input, __gm__ T *output, __gm__ T *gva,
                                        int elements, __gm__ int *magic)
{
    const int64_t aiv_num = GetBlockNum();
    const int64_t aiv_index = GetBlockIdx();

    const int64_t data_offset = aiv_num * SYNC_FLAG_INTERVAL;
    const int64_t flag_offset = aiv_index * SYNC_FLAG_INTERVAL;

    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    int magic_data = *magic;

    __gm__ T *input_gm = (__gm__ T *)input;
    __gm__ T *output_gm = (__gm__ T *)output;

    __gm__ T *gva_data_gm = (__gm__ T *)((__gm__ int32_t *)gva + data_offset);
    __gm__ int32_t *gva_sync_gm = (__gm__ int32_t *)gva;

    __ubuf__ T *tmp_buff = (__ubuf__ T *)(64);

    // data move parameters
    const uint32_t ub_size = UB_DMA_MAX_SIZE;
    uint32_t input_offset, gva_offset, num_per_core;

    // [AllGather Step 1] local input gm -> symmetric mem.
    num_per_core = elements / aiv_num;
    input_offset = aiv_index * num_per_core;
    gva_offset = aiv_index * num_per_core;
    if (aiv_index == aiv_num - 1) {
        num_per_core = elements - num_per_core * aiv_index;
    }
    aclshmemx_mte_put_nbi(gva_data_gm + gva_offset, input_gm + input_offset, tmp_buff, ub_size, num_per_core, my_rank,
                          EVENT_ID0);

    const int64_t core_per_rank = aiv_num / pe_size;
    const int64_t core_rank_idx = aiv_index % core_per_rank;
    const int64_t x = aiv_index / core_per_rank;

    // Sync Ensure Corresponding Tasks Done.
    aclshmem_quiet();

    aclshmemx_signal_op(gva_sync_gm + flag_offset, magic_data, ACLSHMEM_SIGNAL_SET, my_rank);
    aclshmem_signal_wait_until((__gm__ int32_t *)aclshmem_ptr(gva_sync_gm, x) + flag_offset, ACLSHMEM_CMP_EQ, magic_data);

    // [AllGather Step 2] symmetric mem -> local output.
    num_per_core = elements / core_per_rank;
    uint32_t output_offset = x * elements + core_rank_idx * num_per_core;
    gva_offset = core_rank_idx * num_per_core;
    if (core_rank_idx == core_per_rank - 1) {
        num_per_core = elements - num_per_core * core_rank_idx;
    }
    aclshmemx_mte_get_nbi(output_gm + output_offset, gva_data_gm + gva_offset, tmp_buff, ub_size, num_per_core, x,
                          EVENT_ID0);
}

#define ALLGATHER_FUNC_DEF(type)                                                                                   \
    extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void ShmemAllGather_##type(uint64_t ffts_addr, GM_ADDR input, GM_ADDR output,  \
                                                                GM_ADDR gva, int elements, __gm__ int *magic)              \
    {                                                                                                              \
        if (elements * sizeof(type) < 2097152) {                                                                   \
            all_gather_small_data<type>(ffts_addr, (__gm__ type *)input, (__gm__ type *)output, (__gm__ type *)gva, \
                                        elements, magic);                                                          \
        } else {                                                                                                   \
            all_gather_big_data<type>(ffts_addr, (__gm__ type *)input, (__gm__ type *)output, (__gm__ type *)gva,   \
                                      elements, magic);                                                            \
        }                                                                                                          \
    }

#define TYPE_FUNC(fun) fun(int)

TYPE_FUNC(ALLGATHER_FUNC_DEF);

template <class T>
void allgather_demo(uint32_t block_dim, void *stream, uint64_t ffts_addr, uint8_t *input, uint8_t *output, uint8_t *gva,
                    int elements, int *magic)
{
    if (std::is_same<T, int>::value) {
        ShmemAllGather_int<<<block_dim, nullptr, stream>>>(ffts_addr, input, output, gva, elements, magic);
    }
}

template void allgather_demo<int>(uint32_t block_dim, void *stream, uint64_t ffts_addr, uint8_t *input, uint8_t *output,
                                  uint8_t *gva, int elements, int *magic);

template void run_vector_add<int>(int64_t numElements, void *a, void *b, void *c, void *stream);
