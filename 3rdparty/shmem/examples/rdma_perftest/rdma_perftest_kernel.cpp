/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"

constexpr uint32_t MAGIC_VAL = 10;
constexpr uint32_t WARMUP_MESSAGE_LENGTH = 32;

extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_highlevel_put_pingpong_latency(uint64_t fftsConfig, GM_ADDR gva, int message_length) {
    util_set_ffts_config(fftsConfig);
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE);
    AscendC::LocalTensor<uint32_t> ubLocalRead = buf.GetWithOffset<uint32_t>(UB_ALIGN_SIZE / sizeof(uint32_t), 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    uint32_t peer;

    // Warm up
    GM_ADDR warm_addr = gva + rank_size * message_length + WARMUP_MESSAGE_LENGTH * (rank + 1);
    if (rank == 0) {
        peer = 1;
        aclshmem_uint8_put_nbi(warm_addr, warm_addr, WARMUP_MESSAGE_LENGTH, peer);
        while (*(__gm__ uint32_t*)(gva + rank_size * message_length + WARMUP_MESSAGE_LENGTH * (peer + 1)) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + rank_size * message_length + WARMUP_MESSAGE_LENGTH * (peer + 1), sizeof(uint32_t));
            AscendC::GetSystemCycle();
        }
    } else {
        peer = 0;
        while (*(__gm__ uint32_t*)(gva + rank_size * message_length + WARMUP_MESSAGE_LENGTH * (peer + 1)) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + rank_size * message_length + WARMUP_MESSAGE_LENGTH * (peer + 1), sizeof(uint32_t));
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmem_uint8_put_nbi(warm_addr, warm_addr, WARMUP_MESSAGE_LENGTH, peer);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    // Actual test
    GM_ADDR src_addr = gva + rank * message_length;
    if (rank == 0) {
        peer = 1;
        int64_t start = AscendC::GetSystemCycle();
        aclshmem_uint8_put_nbi(src_addr, src_addr, message_length, peer);
        while (*(__gm__ uint32_t*)(gva + message_length * 2 - 8) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + message_length * 2 - 8, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(gva + message_length * 2) = end - start;
    } else {
        peer = 0;
        while (*(__gm__ uint32_t*)(gva + message_length * 1 - 8) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + message_length * 1 - 8, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmem_uint8_put_nbi(src_addr, src_addr, message_length, peer);
    }
}

void rdma_highlevel_put_pingpong_latency_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length) {
    rdma_highlevel_put_pingpong_latency<<<1, nullptr, stream>>>(fftsConfig, gva, message_length);
}

extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_postsend_cost(uint64_t fftsConfig, GM_ADDR gva, int message_length) {
    util_set_ffts_config(fftsConfig);
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint32_t> ubLocal32 = buf.GetWithOffset<uint32_t>(UB_ALIGN_SIZE / sizeof(uint32_t), 0);
    AscendC::LocalTensor<uint64_t> ubLocal64 = buf.GetWithOffset<uint64_t>(UB_ALIGN_SIZE / sizeof(uint64_t), UB_ALIGN_SIZE);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    uint32_t peer;

    // Actual test
    GM_ADDR src_addr = gva + rank * message_length;
    
    if (rank == 0) {
        peer = 1;
        GM_ADDR dest_addr = (GM_ADDR)(aclshmem_roce_ptr(src_addr, peer));
        int64_t start = AscendC::GetSystemCycle();
        for (uint32_t i = 0; i < 500; i++) {
            aclshmemi_roce_write(dest_addr, src_addr, peer, 0, message_length, ubLocal64, ubLocal32, 0);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(gva + message_length * 2) = end - start;
    }
}

void rdma_postsend_cost_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length) {
    rdma_postsend_cost<<<1, nullptr, stream>>>(fftsConfig, gva, message_length);
}

extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_highlevel_put_bw(uint64_t fftsConfig, GM_ADDR gva, int message_length) {
    util_set_ffts_config(fftsConfig);
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    uint32_t peer;

    // Actual test
    GM_ADDR src_addr = gva + rank * message_length;
    if (rank == 0) {
        peer = 1;
        int64_t start = AscendC::GetSystemCycle();
        for (int i = 0; i < 10000; i++) {
            aclshmem_uint8_put_nbi(src_addr, src_addr, message_length, peer);
        }
        aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
        aclshmem_uint8_put_nbi(gva + rank_size * message_length + 8, src_addr, sizeof(uint32_t), peer);
        while (*(__gm__ uint32_t*)(gva + message_length * rank_size + 16) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + message_length * rank_size + 16, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t end = AscendC::GetSystemCycle();
        *(__gm__ int64_t*)(gva + message_length * rank_size) = end - start;
    } else {
        peer = 0;
        while (*(__gm__ uint32_t*)(gva + rank_size * message_length + 8) != peer + MAGIC_VAL) {
            dcci_cachelines(gva + rank_size * message_length + 8, 8);
            AscendC::GetSystemCycle();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmem_uint8_put_nbi(gva + message_length * rank_size + 16, src_addr, sizeof(uint32_t), peer);
    }
}

void rdma_highlevel_put_bw_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length) {
    rdma_highlevel_put_bw<<<1, nullptr, stream>>>(fftsConfig, gva, message_length);
}

extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void rdma_mte_put_bw(uint64_t fftsConfig, GM_ADDR gva, int message_length, int64_t iter) {
    util_set_ffts_config(fftsConfig);
    AscendC::LocalTensor<uint32_t> ubLocal;
    ubLocal.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ubLocal.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
    ubLocal.address_.dataLen = UB_ALIGN_SIZE_64;

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    uint32_t peer;

    // Core 0, RDMA
    if (AscendC::GetBlockIdx() == 0) {
        GM_ADDR src_addr = gva + rank * message_length;
        if (rank == 0) {
            peer = 1;
            int64_t start = AscendC::GetSystemCycle();
            for (int i = 0; i < 10000; i++) {
                aclshmemx_roce_put_nbi(src_addr, src_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), message_length, peer, 0);
            }
            aclshmemx_roce_quiet(peer, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), 0);
            aclshmemx_roce_put_nbi(gva + rank_size * message_length * 2 + 8, src_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), sizeof(int64_t), peer, 0);
            while (*(__gm__ int64_t*)(gva + message_length * rank_size * 2 + 16) != peer + MAGIC_VAL + iter) {
                dcci_cachelines(gva + message_length * rank_size * 2 + 16, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            int64_t end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(gva + message_length * rank_size * 2) = end - start;
        } else {
            peer = 0;
            while (*(__gm__ int64_t*)(gva + rank_size * message_length * 2 + 8) != peer + MAGIC_VAL + iter) {
                dcci_cachelines(gva + rank_size * message_length * 2 + 8, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            aclshmemx_roce_put_nbi(gva + rank_size * message_length * 2 + 16, src_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), sizeof(int64_t), peer, 0);
        }
    } else { // core 1, MTE
        GM_ADDR src_addr = gva + (rank + rank_size) * message_length;
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
        /* CopyUB Config Set */
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
        uint32_t copy_ub_size = device_state->mte_config.ub_size;
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;
        if (rank == 0) {
            peer = 1;
            int64_t start = AscendC::GetSystemCycle();
            for (int i = 0; i < 10000; i++) {
                aclshmemx_mte_put_nbi(src_addr, src_addr, reinterpret_cast<__ubuf__ uint8_t*>(copy_ub), copy_ub_size, message_length, peer, copy_event_id);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            aclshmemx_mte_put_nbi(gva + rank_size * message_length * 2 + 24, src_addr, reinterpret_cast<__ubuf__ uint8_t*>(copy_ub), copy_ub_size, sizeof(uint32_t), peer, copy_event_id);
            while (*(__gm__ uint32_t*)(gva + message_length * rank_size * 2 + 32) != peer + MAGIC_VAL + iter) {
                dcci_cachelines(gva + message_length * rank_size * 2 + 32, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            int64_t end = AscendC::GetSystemCycle();
            *(__gm__ int64_t*)(gva + message_length * rank_size * 2 + 48) = end - start;
        } else {
            peer = 0;
            while (*(__gm__ uint32_t*)(gva + rank_size * message_length * 2 + 24) != peer + MAGIC_VAL + iter) {
                dcci_cachelines(gva + rank_size * message_length * 2 + 24, 8);
                AscendC::GetSystemCycle();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            aclshmemx_mte_put_nbi(gva + rank_size * message_length * 2 + 32, src_addr, reinterpret_cast<__ubuf__ uint8_t*>(copy_ub), copy_ub_size, sizeof(uint32_t), peer, copy_event_id);
        }
    }
}

void rdma_mte_put_bw_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length, int64_t iter) {
    rdma_mte_put_bw<<<2, nullptr, stream>>>(fftsConfig, gva, message_length, iter);
}