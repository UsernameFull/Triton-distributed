/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "combine_kernel.h"
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"
#include "utils/prof/shmemi_prof.h"

#include <type_traits>

#undef inline
#include "opdev/fp16_t.h"
#define inline inline attribute((always_inline))

using namespace AscendC;

using fp16_t = op::fp16_t;

constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t COMBINE_ASSIST_FIELDS = 3;
constexpr int64_t COMBINE_ALIGN_BYTES = 32;
constexpr int64_t COMBINE_SLOT_INT32 = COMBINE_ALIGN_BYTES / static_cast<int64_t>(sizeof(int32_t));
constexpr uint32_t COMBINE_STATUS_UB_OFFSET = 32;
constexpr int32_t COMBINE_STATUS_READY = 1;

ACLSHMEM_DEVICE int64_t CombineAlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

ACLSHMEM_DEVICE int64_t CombineSegmentBegin(__gm__ int32_t *ep_recv_count, int64_t segment)
{
    return (segment == 0) ? 0 : ep_recv_count[segment - 1];
}

ACLSHMEM_DEVICE int64_t CombineSegmentEnd(__gm__ int32_t *ep_recv_count, int64_t segment)
{
    return ep_recv_count[segment];
}

template <typename T>
ACLSHMEM_DEVICE void combine_classic_once(__gm__ T *expand_x, __gm__ int32_t *assist_info_for_combine,
                                          __gm__ int32_t *ep_recv_count, __gm__ int32_t *expert_ids,
                                          __gm__ float *expert_scales, __gm__ T *x_out, __gm__ uint8_t *shmem_window,
                                          int bs, int h, int k, int moe_expert_num, int comm_frame_id,
                                          bool enable_prof)
{
    const int64_t aiv_index = GetBlockIdx();
    const int64_t pe_size = aclshmem_n_pes();
    const bool active_core = aiv_index < pe_size;
    const int64_t local_expert_num = moe_expert_num / pe_size;
    const int64_t slot_num = bs * k;
    const int64_t data_stride =
        CombineAlignUp(h * static_cast<int64_t>(sizeof(T)), COMBINE_ALIGN_BYTES) / static_cast<int64_t>(sizeof(T));
    const int64_t data_bytes = slot_num * data_stride * static_cast<int64_t>(sizeof(T));

    __gm__ T *data_base = (__gm__ T *)shmem_window;
    __gm__ int32_t *status_base = (__gm__ int32_t *)(shmem_window + data_bytes);
    __ubuf__ T *tmp_buff = (__ubuf__ T *)(64);
    __ubuf__ int32_t *status_ub = (__ubuf__ int32_t *)(static_cast<uint64_t>(COMBINE_STATUS_UB_OFFSET));
    for (int64_t status_idx = 0; status_idx < COMBINE_SLOT_INT32; ++status_idx) {
        status_ub[status_idx] = COMBINE_STATUS_READY;
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);

    // Stage 1: send expert outputs back to the original token owner using dispatch assist metadata.
    if (enable_prof) {
        SHMEMI_PROF_START(comm_frame_id);
    }
    if (active_core) {
        const int64_t target_src_rank = aiv_index;
        for (int64_t local_expert = 0; local_expert < local_expert_num; ++local_expert) {
            const int64_t segment = local_expert * pe_size + target_src_rank;
            const int64_t begin = CombineSegmentBegin(ep_recv_count, segment);
            const int64_t end = CombineSegmentEnd(ep_recv_count, segment);
            for (int64_t i = begin; i < end; ++i) {
                const int32_t src_rank = assist_info_for_combine[i * COMBINE_ASSIST_FIELDS];
                const int32_t token_id = assist_info_for_combine[i * COMBINE_ASSIST_FIELDS + 1];
                const int32_t topk_id = assist_info_for_combine[i * COMBINE_ASSIST_FIELDS + 2];
                const int64_t slot = token_id * k + topk_id;

                aclshmemx_mte_put_nbi(data_base + slot * data_stride, expand_x + i * h, tmp_buff, UB_DMA_MAX_SIZE, h,
                                      src_rank, EVENT_ID0);
                aclshmem_quiet();
                aclshmemx_mte_put_nbi(status_base + slot * COMBINE_SLOT_INT32, status_ub,
                                      static_cast<uint32_t>(COMBINE_SLOT_INT32), src_rank, EVENT_ID0);
                aclshmem_quiet();
            }
        }
    }

    aclshmem_quiet();
    if (enable_prof) {
        SHMEMI_PROF_END(comm_frame_id);
    }

    // Stage 2: wait for every topK result of each local token and reduce with expert_scales.
    for (int64_t token_id = aiv_index; token_id < bs; token_id += GetBlockNum()) {
        for (int64_t topk_id = 0; topk_id < k; ++topk_id) {
            const int64_t slot = token_id * k + topk_id;
            for (int64_t status_idx = 0; status_idx < COMBINE_SLOT_INT32; ++status_idx) {
                aclshmem_signal_wait_until(status_base + slot * COMBINE_SLOT_INT32 + status_idx, ACLSHMEM_CMP_EQ,
                                           COMBINE_STATUS_READY);
            }
        }

        for (int64_t j = 0; j < h; ++j) {
            float acc = 0.0F;
            for (int64_t topk_id = 0; topk_id < k; ++topk_id) {
                const int64_t slot = token_id * k + topk_id;
                const int32_t expert_id = expert_ids[slot];
                if (expert_id >= 0 && expert_id < moe_expert_num) {
                    acc += static_cast<float>(data_base[slot * data_stride + j]) * expert_scales[slot];
                }
            }
            x_out[token_id * h + j] = static_cast<T>(acc);
        }

        for (int64_t topk_id = 0; topk_id < k; ++topk_id) {
            const int64_t slot = token_id * k + topk_id;
            for (int64_t status_idx = 0; status_idx < COMBINE_SLOT_INT32; ++status_idx) {
                status_base[slot * COMBINE_SLOT_INT32 + status_idx] = 0;
            }
        }
    }

    aclshmemi_barrier_core_soft();
}

template <typename T>
ACLSHMEM_DEVICE void combine_classic(uint64_t fftsAddr, __gm__ T *expand_x,
                                     __gm__ int32_t *assist_info_for_combine, __gm__ int32_t *ep_recv_count,
                                     __gm__ int32_t *expert_ids, __gm__ float *expert_scales, __gm__ T *x_out,
                                     __gm__ uint8_t *shmem_window, int bs, int h, int k, int moe_expert_num,
                                     int magic, int perf_mode, int full_frame_id, int comm_frame_id,
                                     int warmup_count, int loop_count)
{
    (void)magic;
    (void)warmup_count;
    (void)loop_count;
    util_set_ffts_config(fftsAddr);

    const bool enable_prof = perf_mode != 0;
    if (enable_prof) {
        SHMEMI_PROF_START(full_frame_id);
    }

    combine_classic_once<T>(expand_x, assist_info_for_combine, ep_recv_count, expert_ids, expert_scales, x_out,
                            shmem_window, bs, h, k, moe_expert_num, comm_frame_id, enable_prof);

    if (enable_prof) {
        SHMEMI_PROF_END(full_frame_id);
    }
}

#define COMBINE_FUNC_DEF(type)                                                                                       \
    extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ShmemCombine_##type(                        \
        uint64_t fftsAddr, GM_ADDR expand_x, GM_ADDR assist_info_for_combine, GM_ADDR ep_recv_count,                 \
        GM_ADDR expert_ids, GM_ADDR expert_scales, GM_ADDR x_out, GM_ADDR shmem_window, int bs, int h, int k,         \
        int moe_expert_num, int magic, int perf_mode, int full_frame_id, int comm_frame_id, int warmup_count,         \
        int loop_count)                                                                                                \
    {                                                                                                                 \
        combine_classic<type>(fftsAddr, (__gm__ type *)expand_x, (__gm__ int32_t *)assist_info_for_combine,           \
                              (__gm__ int32_t *)ep_recv_count, (__gm__ int32_t *)expert_ids,                          \
                              (__gm__ float *)expert_scales, (__gm__ type *)x_out, (__gm__ uint8_t *)shmem_window,    \
                              bs, h, k, moe_expert_num, magic, perf_mode, full_frame_id, comm_frame_id, warmup_count, \
                              loop_count);                                                                            \
    }

COMBINE_FUNC_DEF(int32_t);
COMBINE_FUNC_DEF(float16_t);

template <class T>
void combine_demo(uint32_t block_dim, void *stream, uint64_t fftsAddr, uint8_t *expand_x,
                  int32_t *assist_info_for_combine, int32_t *ep_recv_count, int32_t *expert_ids,
                  float *expert_scales, uint8_t *x_out, uint8_t *shmem_window, int bs, int h, int k,
                  int moe_expert_num, int magic, int perf_mode, int full_frame_id, int comm_frame_id,
                  int warmup_count, int loop_count)
{
    if (std::is_same<T, int32_t>::value || std::is_same<T, int>::value) {
        ShmemCombine_int32_t<<<block_dim, nullptr, stream>>>(
            fftsAddr, expand_x, reinterpret_cast<uint8_t *>(assist_info_for_combine),
            reinterpret_cast<uint8_t *>(ep_recv_count), reinterpret_cast<uint8_t *>(expert_ids),
            reinterpret_cast<uint8_t *>(expert_scales), x_out, shmem_window, bs, h, k, moe_expert_num, magic,
            perf_mode, full_frame_id, comm_frame_id, warmup_count, loop_count);
    } else if (std::is_same<T, fp16_t>::value) {
        ShmemCombine_float16_t<<<block_dim, nullptr, stream>>>(
            fftsAddr, expand_x, reinterpret_cast<uint8_t *>(assist_info_for_combine),
            reinterpret_cast<uint8_t *>(ep_recv_count), reinterpret_cast<uint8_t *>(expert_ids),
            reinterpret_cast<uint8_t *>(expert_scales), x_out, shmem_window, bs, h, k, moe_expert_num, magic,
            perf_mode, full_frame_id, comm_frame_id, warmup_count, loop_count);
    }
}

template void combine_demo<int32_t>(uint32_t block_dim, void *stream, uint64_t fftsAddr, uint8_t *expand_x,
                                    int32_t *assist_info_for_combine, int32_t *ep_recv_count, int32_t *expert_ids,
                                    float *expert_scales, uint8_t *x_out, uint8_t *shmem_window, int bs, int h, int k,
                                    int moe_expert_num, int magic, int perf_mode, int full_frame_id,
                                    int comm_frame_id, int warmup_count, int loop_count);
template void combine_demo<fp16_t>(uint32_t block_dim, void *stream, uint64_t fftsAddr, uint8_t *expand_x,
                                   int32_t *assist_info_for_combine, int32_t *ep_recv_count, int32_t *expert_ids,
                                   float *expert_scales, uint8_t *x_out, uint8_t *shmem_window, int bs, int h, int k,
                                   int moe_expert_num, int magic, int perf_mode, int full_frame_id, int comm_frame_id,
                                   int warmup_count, int loop_count);
