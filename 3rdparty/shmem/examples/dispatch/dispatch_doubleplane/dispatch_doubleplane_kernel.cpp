/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dispatch_kernel.h"
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"
#include "utils/prof/shmemi_prof.h"

#include <type_traits>

#undef inline
#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"
#define inline inline attribute((always_inline))

using namespace AscendC;

using fp16_t = op::fp16_t;
using bf16_t = op::bfloat16;

constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr uint32_t DIRECT_UB_OFFSET = 64;
constexpr uint32_t SDMA_UB_SIZE = 64;
constexpr uint32_t SDMA_UB_OFFSET = 191 * 1024;
constexpr uint32_t DIRECT_EVENT_ID = EVENT_ID1;
constexpr uint32_t SDMA_EVENT_ID = EVENT_ID0;
constexpr int64_t SDMA_ISSUE_LIMIT = 256;
constexpr uint64_t MIN_SDMA_BYTES = 2 * 1024 * 1024;
constexpr int64_t DISPATCH_ASSIST_FIELDS = 3;
constexpr int64_t DISPATCH_ALIGN_BYTES = 32;
constexpr int64_t DISPATCH_SLOT_INT32 = DISPATCH_ALIGN_BYTES / static_cast<int64_t>(sizeof(int32_t));
constexpr int32_t DISPATCH_COUNT_INIT = -1;
constexpr uint32_t DISPATCH_STATUS_UB_OFFSET = 32;
constexpr int32_t DISPATCH_READY_VALUE = 1;
constexpr int32_t DISPATCH_COUNT_READY = 1;

ACLSHMEM_DEVICE int64_t DispatchAlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

ACLSHMEM_DEVICE bool DispatchUseSdma(int64_t peer_rank, int64_t my_rank, int64_t token_count, int h, int64_t elem_size,
                                     uint64_t threshold_num, uint64_t threshold_den)
{
    if (peer_rank == my_rank || token_count <= 0 || threshold_den == 0) {
        return false;
    }
    const uint64_t bytes = static_cast<uint64_t>(token_count) * static_cast<uint64_t>(h) *
                           static_cast<uint64_t>(elem_size);
    if (bytes < MIN_SDMA_BYTES) {
        return false;
    }
    return bytes * threshold_den > threshold_num;
}

ACLSHMEM_DEVICE uint64_t DispatchThresholdDen(int64_t pe_size, int64_t local_expert_num)
{
    const int64_t remote_rank_num = pe_size > 1 ? pe_size - 1 : 1;
    return static_cast<uint64_t>(remote_rank_num) * static_cast<uint64_t>(local_expert_num);
}

ACLSHMEM_DEVICE void DispatchFillReadyStatus(__ubuf__ int32_t *status_ub)
{
    for (int64_t status_idx = 0; status_idx < DISPATCH_SLOT_INT32; ++status_idx) {
        status_ub[status_idx] = DISPATCH_READY_VALUE;
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
}

ACLSHMEM_DEVICE void DispatchFillCountStatus(__ubuf__ int32_t *status_ub, int32_t count)
{
    status_ub[0] = DISPATCH_COUNT_READY;
    status_ub[1] = count;
    for (int64_t status_idx = 2; status_idx < DISPATCH_SLOT_INT32; ++status_idx) {
        status_ub[status_idx] = 0;
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
}

ACLSHMEM_DEVICE void SignalDispatchAssist(__gm__ int32_t *assist_base, int64_t global_slot, int32_t my_rank,
                                          int32_t token_id, int32_t topk_id, int64_t dst_rank)
{
    aclshmemx_signal_op(assist_base + global_slot * DISPATCH_SLOT_INT32, my_rank, ACLSHMEM_SIGNAL_SET, dst_rank);
    aclshmemx_signal_op(assist_base + global_slot * DISPATCH_SLOT_INT32 + 1, token_id, ACLSHMEM_SIGNAL_SET, dst_rank);
    aclshmemx_signal_op(assist_base + global_slot * DISPATCH_SLOT_INT32 + 2, topk_id, ACLSHMEM_SIGNAL_SET, dst_rank);
}

template <typename T>
ACLSHMEM_DEVICE void dispatch_classic(uint64_t fftsAddr, __gm__ T *x, __gm__ int32_t *expert_ids,
                                      __gm__ T *expand_x, __gm__ int32_t *assist_info_for_combine,
                                      __gm__ int32_t *ep_recv_count, __gm__ int32_t *expert_token_nums,
                                      __gm__ uint8_t *shmem_window, int bs, int h, int k, int moe_expert_num,
                                      int magic, int perf_mode, int full_frame_id, int comm_frame_id)
{
    (void)magic;
    util_set_ffts_config(fftsAddr);

    const int64_t aiv_index = GetBlockIdx();
    const int64_t my_rank = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();
    // The host passes moe_expert_num = expert_per_pe * pe_size. Guard the kernel anyway so
    // any mismatched standalone launch cannot silently truncate local_expert_num.
    if (pe_size <= 0 || moe_expert_num <= 0 || (static_cast<int64_t>(moe_expert_num) % pe_size) != 0) {
        return;
    }
    const int64_t local_expert_num = moe_expert_num / pe_size;
    if (local_expert_num > DISPATCH_MAX_LOCAL_EXPERT_NUM) {
        return;
    }
    const bool active_core = aiv_index < pe_size;
    const int64_t max_tokens_per_segment = bs * k;
    const int64_t segment_num = pe_size * local_expert_num;
    const int64_t total_slots = segment_num * max_tokens_per_segment;

    const int64_t payload_stride =
        DispatchAlignUp(h * static_cast<int64_t>(sizeof(T)), DISPATCH_ALIGN_BYTES) / static_cast<int64_t>(sizeof(T));
    const int64_t payload_bytes = total_slots * payload_stride * static_cast<int64_t>(sizeof(T));
    const int64_t assist_bytes = total_slots * DISPATCH_SLOT_INT32 * static_cast<int64_t>(sizeof(int32_t));
    const int64_t ready_bytes = total_slots * DISPATCH_SLOT_INT32 * static_cast<int64_t>(sizeof(int32_t));

    __gm__ T *payload_base = (__gm__ T *)shmem_window;
    __gm__ int32_t *assist_base = (__gm__ int32_t *)(shmem_window + payload_bytes);
    __gm__ int32_t *ready_base = (__gm__ int32_t *)(shmem_window + payload_bytes + assist_bytes);
    __gm__ int32_t *count_base = (__gm__ int32_t *)(shmem_window + payload_bytes + assist_bytes + ready_bytes);

    __ubuf__ T *direct_tmp = (__ubuf__ T *)(static_cast<uint64_t>(DIRECT_UB_OFFSET));
    __ubuf__ T *sdma_tmp = (__ubuf__ T *)(static_cast<uint64_t>(SDMA_UB_OFFSET));
    __ubuf__ int32_t *status_ub = (__ubuf__ int32_t *)(static_cast<uint64_t>(DISPATCH_STATUS_UB_OFFSET));

    // Stage 1: route every local token/topK pair to the rank that owns the selected expert.
    if (perf_mode != 0) {
        SHMEMI_PROF_START(full_frame_id);
        SHMEMI_PROF_START(comm_frame_id);
    }
    if (active_core) {
        const int64_t dst_rank = aiv_index;
        int32_t segment_counts[DISPATCH_MAX_LOCAL_EXPERT_NUM];
        int32_t slot_offsets[DISPATCH_MAX_LOCAL_EXPERT_NUM];
        int32_t sdma_flags[DISPATCH_MAX_LOCAL_EXPERT_NUM];
        for (int64_t dst_local_expert = 0; dst_local_expert < local_expert_num; ++dst_local_expert) {
            segment_counts[dst_local_expert] = 0;
            slot_offsets[dst_local_expert] = 0;
            sdma_flags[dst_local_expert] = 0;
        }

        uint64_t remote_tokens = 0;
        for (int64_t flat = 0; flat < bs * k; ++flat) {
            const int64_t expert_id = expert_ids[flat];
            const int64_t route_dst_rank = expert_id / local_expert_num;
            if (route_dst_rank != my_rank) {
                ++remote_tokens;
            }
            if (route_dst_rank != dst_rank) {
                continue;
            }
            const int64_t dst_local_expert = expert_id % local_expert_num;
            ++segment_counts[dst_local_expert];
        }

        const uint64_t threshold_num =
            remote_tokens * static_cast<uint64_t>(h) * static_cast<uint64_t>(sizeof(T));
        const uint64_t threshold_den = DispatchThresholdDen(pe_size, local_expert_num);
        bool has_sdma = false;
        for (int64_t dst_local_expert = 0; dst_local_expert < local_expert_num; ++dst_local_expert) {
            const int64_t token_count = segment_counts[dst_local_expert];
            const bool use_sdma =
                DispatchUseSdma(dst_rank, my_rank, token_count, h, static_cast<int64_t>(sizeof(T)), threshold_num,
                                threshold_den);
            sdma_flags[dst_local_expert] = use_sdma ? 1 : 0;
            has_sdma = has_sdma || use_sdma;
        }

        // SDMA payload phase. Control signals are emitted only after all SDMA payloads are visible.
        int64_t sdma_outstanding = 0;
        for (int64_t flat = 0; flat < bs * k; ++flat) {
            const int64_t expert_id = expert_ids[flat];
            const int64_t route_dst_rank = expert_id / local_expert_num;
            if (route_dst_rank != dst_rank) {
                continue;
            }
            const int64_t dst_local_expert = expert_id % local_expert_num;
            if (sdma_flags[dst_local_expert] == 0) {
                continue;
            }

            const int64_t token_id = flat / k;
            const int64_t slot = slot_offsets[dst_local_expert]++;
            const int64_t data_block = my_rank * local_expert_num + dst_local_expert;
            const int64_t global_slot = data_block * max_tokens_per_segment + slot;

            aclshmemx_sdma_put_nbi(payload_base + global_slot * payload_stride, x + token_id * h, sdma_tmp,
                                   SDMA_UB_SIZE, h, dst_rank, SDMA_EVENT_ID);
            ++sdma_outstanding;
            if (sdma_outstanding >= SDMA_ISSUE_LIMIT) {
                aclshmemx_sdma_quiet(sdma_tmp, SDMA_UB_SIZE, SDMA_EVENT_ID);
                sdma_outstanding = 0;
            }
        }
        if (has_sdma && sdma_outstanding > 0) {
            aclshmemx_sdma_quiet(sdma_tmp, SDMA_UB_SIZE, SDMA_EVENT_ID);
        }

        // Direct payload phase keeps the original per-token payload quiet while batching control-plane writes.
        for (int64_t dst_local_expert = 0; dst_local_expert < local_expert_num; ++dst_local_expert) {
            slot_offsets[dst_local_expert] = 0;
        }
        for (int64_t flat = 0; flat < bs * k; ++flat) {
            const int64_t expert_id = expert_ids[flat];
            const int64_t route_dst_rank = expert_id / local_expert_num;
            if (route_dst_rank != dst_rank) {
                continue;
            }
            const int64_t dst_local_expert = expert_id % local_expert_num;
            if (sdma_flags[dst_local_expert] != 0) {
                continue;
            }

            const int64_t token_id = flat / k;
            const int64_t slot = slot_offsets[dst_local_expert]++;
            const int64_t data_block = my_rank * local_expert_num + dst_local_expert;
            const int64_t global_slot = data_block * max_tokens_per_segment + slot;

            aclshmemx_mte_put_nbi(payload_base + global_slot * payload_stride, x + token_id * h, direct_tmp,
                                  UB_DMA_MAX_SIZE, h, dst_rank, DIRECT_EVENT_ID);
            aclshmem_quiet();
        }

        for (int64_t dst_local_expert = 0; dst_local_expert < local_expert_num; ++dst_local_expert) {
            slot_offsets[dst_local_expert] = 0;
        }
        bool wrote_direct_assist = false;
        for (int64_t flat = 0; flat < bs * k; ++flat) {
            const int64_t expert_id = expert_ids[flat];
            const int64_t route_dst_rank = expert_id / local_expert_num;
            if (route_dst_rank != dst_rank) {
                continue;
            }
            const int64_t dst_local_expert = expert_id % local_expert_num;
            if (sdma_flags[dst_local_expert] != 0) {
                continue;
            }

            const int64_t token_id = flat / k;
            const int64_t topk_id = flat % k;
            const int64_t slot = slot_offsets[dst_local_expert]++;
            const int64_t data_block = my_rank * local_expert_num + dst_local_expert;
            const int64_t global_slot = data_block * max_tokens_per_segment + slot;

            SignalDispatchAssist(assist_base, global_slot, static_cast<int32_t>(my_rank),
                                 static_cast<int32_t>(token_id), static_cast<int32_t>(topk_id), dst_rank);
            wrote_direct_assist = true;
        }
        if (wrote_direct_assist) {
            aclshmem_quiet();

            for (int64_t dst_local_expert = 0; dst_local_expert < local_expert_num; ++dst_local_expert) {
                slot_offsets[dst_local_expert] = 0;
            }
            DispatchFillReadyStatus(status_ub);
            bool wrote_direct_ready = false;
            for (int64_t flat = 0; flat < bs * k; ++flat) {
                const int64_t expert_id = expert_ids[flat];
                const int64_t route_dst_rank = expert_id / local_expert_num;
                if (route_dst_rank != dst_rank) {
                    continue;
                }
                const int64_t dst_local_expert = expert_id % local_expert_num;
                if (sdma_flags[dst_local_expert] != 0) {
                    continue;
                }

                const int64_t slot = slot_offsets[dst_local_expert]++;
                const int64_t data_block = my_rank * local_expert_num + dst_local_expert;
                const int64_t global_slot = data_block * max_tokens_per_segment + slot;

                aclshmemx_mte_put_nbi(ready_base + global_slot * DISPATCH_SLOT_INT32, status_ub,
                                      static_cast<uint32_t>(DISPATCH_SLOT_INT32), dst_rank, DIRECT_EVENT_ID);
                wrote_direct_ready = true;
            }
            if (wrote_direct_ready) {
                aclshmem_quiet();
            }
        }

        for (int64_t dst_local_expert = 0; dst_local_expert < local_expert_num; ++dst_local_expert) {
            if (sdma_flags[dst_local_expert] != 0) {
                continue;
            }
            const int64_t status_segment = dst_local_expert * pe_size + my_rank;
            DispatchFillCountStatus(status_ub, segment_counts[dst_local_expert]);
            aclshmemx_mte_put_nbi(count_base + status_segment * DISPATCH_SLOT_INT32, status_ub,
                                  static_cast<uint32_t>(DISPATCH_SLOT_INT32), dst_rank, DIRECT_EVENT_ID);
            aclshmem_quiet();
        }

        for (int64_t dst_local_expert = 0; dst_local_expert < local_expert_num; ++dst_local_expert) {
            slot_offsets[dst_local_expert] = 0;
        }
        bool wrote_sdma_assist = false;
        for (int64_t flat = 0; flat < bs * k; ++flat) {
            const int64_t expert_id = expert_ids[flat];
            const int64_t route_dst_rank = expert_id / local_expert_num;
            if (route_dst_rank != dst_rank) {
                continue;
            }
            const int64_t dst_local_expert = expert_id % local_expert_num;
            if (sdma_flags[dst_local_expert] == 0) {
                continue;
            }

            const int64_t token_id = flat / k;
            const int64_t topk_id = flat % k;
            const int64_t slot = slot_offsets[dst_local_expert]++;
            const int64_t data_block = my_rank * local_expert_num + dst_local_expert;
            const int64_t global_slot = data_block * max_tokens_per_segment + slot;

            SignalDispatchAssist(assist_base, global_slot, static_cast<int32_t>(my_rank),
                                 static_cast<int32_t>(token_id), static_cast<int32_t>(topk_id), dst_rank);
            wrote_sdma_assist = true;
        }
        if (wrote_sdma_assist) {
            aclshmem_quiet();

            for (int64_t dst_local_expert = 0; dst_local_expert < local_expert_num; ++dst_local_expert) {
                slot_offsets[dst_local_expert] = 0;
            }
            DispatchFillReadyStatus(status_ub);
            bool wrote_sdma_ready = false;
            for (int64_t flat = 0; flat < bs * k; ++flat) {
                const int64_t expert_id = expert_ids[flat];
                const int64_t route_dst_rank = expert_id / local_expert_num;
                if (route_dst_rank != dst_rank) {
                    continue;
                }
                const int64_t dst_local_expert = expert_id % local_expert_num;
                if (sdma_flags[dst_local_expert] == 0) {
                    continue;
                }

                const int64_t slot = slot_offsets[dst_local_expert]++;
                const int64_t data_block = my_rank * local_expert_num + dst_local_expert;
                const int64_t global_slot = data_block * max_tokens_per_segment + slot;

                aclshmemx_mte_put_nbi(ready_base + global_slot * DISPATCH_SLOT_INT32, status_ub,
                                      static_cast<uint32_t>(DISPATCH_SLOT_INT32), dst_rank, DIRECT_EVENT_ID);
                wrote_sdma_ready = true;
            }
            if (wrote_sdma_ready) {
                aclshmem_quiet();
            }
        }

        for (int64_t dst_local_expert = 0; dst_local_expert < local_expert_num; ++dst_local_expert) {
            if (sdma_flags[dst_local_expert] == 0) {
                continue;
            }
            const int64_t status_segment = dst_local_expert * pe_size + my_rank;
            DispatchFillCountStatus(status_ub, segment_counts[dst_local_expert]);
            aclshmemx_mte_put_nbi(count_base + status_segment * DISPATCH_SLOT_INT32, status_ub,
                                  static_cast<uint32_t>(DISPATCH_SLOT_INT32), dst_rank, DIRECT_EVENT_ID);
            aclshmem_quiet();
        }

        (void)threshold_num;
        (void)threshold_den;
    }

    aclshmem_quiet();
    if (perf_mode != 0) {
        SHMEMI_PROF_END(comm_frame_id);
    }
    aclshmemi_barrier_core_soft();

    // Stage 2: wait for all source ranks, then build cumulative receive counts.
    if (aiv_index == 0) {
        int32_t running = 0;
        for (int64_t local_expert = 0; local_expert < local_expert_num; ++local_expert) {
            int32_t expert_count = 0;
            for (int64_t src_rank = 0; src_rank < pe_size; ++src_rank) {
                const int64_t segment = local_expert * pe_size + src_rank;
                aclshmem_signal_wait_until(count_base + segment * DISPATCH_SLOT_INT32, ACLSHMEM_CMP_EQ,
                                           DISPATCH_COUNT_READY);
                const int32_t count = count_base[segment * DISPATCH_SLOT_INT32 + 1];
                running += count;
                expert_count += count;
                ep_recv_count[segment] = running;
            }
            expert_token_nums[local_expert] = expert_count;
        }
    }

    aclshmemi_barrier_core_soft();

    // Stage 3: compact local window into classic dispatch outputs ordered by (local expert, source rank).
    // Keep compaction on core 0 so prefix boundaries and output writes are observed in one deterministic order.
    if (aiv_index == 0) {
        for (int64_t local_expert = 0; local_expert < local_expert_num; ++local_expert) {
            for (int64_t src_rank = 0; src_rank < pe_size; ++src_rank) {
                const int64_t segment = local_expert * pe_size + src_rank;
                const int64_t begin = (segment == 0) ? 0 : ep_recv_count[segment - 1];
                const int64_t count = count_base[segment * DISPATCH_SLOT_INT32 + 1];
                const int64_t data_block = src_rank * local_expert_num + local_expert;
                for (int64_t i = 0; i < count; ++i) {
                    const int64_t global_slot = data_block * max_tokens_per_segment + i;
                    for (int64_t ready_idx = 0; ready_idx < DISPATCH_SLOT_INT32; ++ready_idx) {
                        aclshmem_signal_wait_until(ready_base + global_slot * DISPATCH_SLOT_INT32 + ready_idx,
                                                   ACLSHMEM_CMP_EQ, DISPATCH_READY_VALUE);
                    }
                    for (int64_t j = 0; j < h; ++j) {
                        expand_x[(begin + i) * h + j] = payload_base[global_slot * payload_stride + j];
                    }
                    for (int64_t j = 0; j < DISPATCH_ASSIST_FIELDS; ++j) {
                        assist_info_for_combine[(begin + i) * DISPATCH_ASSIST_FIELDS + j] =
                            assist_base[global_slot * DISPATCH_SLOT_INT32 + j];
                    }
                    for (int64_t ready_idx = 0; ready_idx < DISPATCH_SLOT_INT32; ++ready_idx) {
                        ready_base[global_slot * DISPATCH_SLOT_INT32 + ready_idx] = 0;
                    }
                }
                for (int64_t count_idx = 0; count_idx < DISPATCH_SLOT_INT32; ++count_idx) {
                    count_base[segment * DISPATCH_SLOT_INT32 + count_idx] = DISPATCH_COUNT_INIT;
                }
            }
        }
    }

    aclshmemi_barrier_core_soft();
    if (perf_mode != 0) {
        SHMEMI_PROF_END(full_frame_id);
    }
}

#define DISPATCH_FUNC_DEF(type)                                                                                       \
    extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ShmemDispatch_##type(                        \
        uint64_t fftsAddr, GM_ADDR x, GM_ADDR expert_ids, GM_ADDR expand_x, GM_ADDR assist_info_for_combine,           \
        GM_ADDR ep_recv_count, GM_ADDR expert_token_nums, GM_ADDR shmem_window, int bs, int h, int k,                  \
        int moe_expert_num, int magic, int perf_mode, int full_frame_id, int comm_frame_id, int warmup_count,          \
        int loop_count)                                                                                                 \
    {                                                                                                                  \
        (void)warmup_count;                                                                                            \
        (void)loop_count;                                                                                              \
        dispatch_classic<type>(fftsAddr, (__gm__ type *)x, (__gm__ int32_t *)expert_ids, (__gm__ type *)expand_x,       \
                               (__gm__ int32_t *)assist_info_for_combine, (__gm__ int32_t *)ep_recv_count,              \
                               (__gm__ int32_t *)expert_token_nums, (__gm__ uint8_t *)shmem_window, bs, h, k,           \
                               moe_expert_num, magic, perf_mode, full_frame_id, comm_frame_id);                        \
    }

DISPATCH_FUNC_DEF(int32_t);
DISPATCH_FUNC_DEF(float16_t);
DISPATCH_FUNC_DEF(bfloat16_t);

template <class T>
void dispatch_demo(uint32_t block_dim, void *stream, uint64_t fftsAddr, uint8_t *x, int32_t *expert_ids,
                   uint8_t *expand_x, int32_t *assist_info_for_combine, int32_t *ep_recv_count,
                   int32_t *expert_token_nums, uint8_t *shmem_window, int bs, int h, int k, int moe_expert_num,
                   int magic, int perf_mode, int full_frame_id, int comm_frame_id, int warmup_count, int loop_count)
{
    if (std::is_same<T, int32_t>::value || std::is_same<T, int>::value) {
        ShmemDispatch_int32_t<<<block_dim, nullptr, stream>>>(
            fftsAddr, x, reinterpret_cast<uint8_t *>(expert_ids), expand_x,
            reinterpret_cast<uint8_t *>(assist_info_for_combine), reinterpret_cast<uint8_t *>(ep_recv_count),
            reinterpret_cast<uint8_t *>(expert_token_nums), shmem_window, bs, h, k, moe_expert_num, magic,
            perf_mode, full_frame_id, comm_frame_id, warmup_count, loop_count);
    } else if (std::is_same<T, fp16_t>::value) {
        ShmemDispatch_float16_t<<<block_dim, nullptr, stream>>>(
            fftsAddr, x, reinterpret_cast<uint8_t *>(expert_ids), expand_x,
            reinterpret_cast<uint8_t *>(assist_info_for_combine), reinterpret_cast<uint8_t *>(ep_recv_count),
            reinterpret_cast<uint8_t *>(expert_token_nums), shmem_window, bs, h, k, moe_expert_num, magic,
            perf_mode, full_frame_id, comm_frame_id, warmup_count, loop_count);
    } else if (std::is_same<T, bf16_t>::value) {
        ShmemDispatch_bfloat16_t<<<block_dim, nullptr, stream>>>(
            fftsAddr, x, reinterpret_cast<uint8_t *>(expert_ids), expand_x,
            reinterpret_cast<uint8_t *>(assist_info_for_combine), reinterpret_cast<uint8_t *>(ep_recv_count),
            reinterpret_cast<uint8_t *>(expert_token_nums), shmem_window, bs, h, k, moe_expert_num, magic,
            perf_mode, full_frame_id, comm_frame_id, warmup_count, loop_count);
    }
}

template void dispatch_demo<int32_t>(uint32_t block_dim, void *stream, uint64_t fftsAddr, uint8_t *x,
                                     int32_t *expert_ids, uint8_t *expand_x, int32_t *assist_info_for_combine,
                                     int32_t *ep_recv_count, int32_t *expert_token_nums, uint8_t *shmem_window,
                                     int bs, int h, int k, int moe_expert_num, int magic, int perf_mode,
                                     int full_frame_id, int comm_frame_id, int warmup_count, int loop_count);
template void dispatch_demo<fp16_t>(uint32_t block_dim, void *stream, uint64_t fftsAddr, uint8_t *x,
                                    int32_t *expert_ids, uint8_t *expand_x, int32_t *assist_info_for_combine,
                                    int32_t *ep_recv_count, int32_t *expert_token_nums, uint8_t *shmem_window,
                                    int bs, int h, int k, int moe_expert_num, int magic, int perf_mode,
                                    int full_frame_id, int comm_frame_id, int warmup_count, int loop_count);
template void dispatch_demo<bf16_t>(uint32_t block_dim, void *stream, uint64_t fftsAddr, uint8_t *x,
                                    int32_t *expert_ids, uint8_t *expand_x, int32_t *assist_info_for_combine,
                                    int32_t *ep_recv_count, int32_t *expert_token_nums, uint8_t *shmem_window,
                                    int bs, int h, int k, int moe_expert_num, int magic, int perf_mode,
                                    int full_frame_id, int comm_frame_id, int warmup_count, int loop_count);
