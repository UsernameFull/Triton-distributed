/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "shmem.h"

constexpr uint64_t MESSAGE_SIZE = 64;

extern "C" __global__ __aicore__ void PutNotifyWaitTest(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;

    if ASCEND_IS_AIV {
        int64_t my_pe = aclshmem_my_pe();
        int64_t n_pes = aclshmem_n_pes();

        // Define temporary UB buffer for SDMA operations
        constexpr uint32_t ub_offset = 1024;
        constexpr uint32_t ub_size = 64;  // 64B for temporary buffer
        __ubuf__ uint8_t *tmp_buff = reinterpret_cast<__ubuf__ uint8_t *>(uint64_t(ub_offset));

        uint32_t data_length = static_cast<uint32_t>(MESSAGE_SIZE);
        const auto cur_block_idx = AscendC::GetBlockIdx();
        const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint64_t base_per_core = data_length / comm_block_dim;
        uint64_t extra_bytes = data_length % comm_block_dim;
        uint64_t data_offset = 0;
        if (cur_block_idx < extra_bytes) {
            data_offset = cur_block_idx * (base_per_core + 1);
        } else {
            data_offset = extra_bytes * (base_per_core + 1) + (cur_block_idx - extra_bytes) * base_per_core;
        }
        if (cur_block_idx < extra_bytes) {
            base_per_core += 1;
        }
        if (base_per_core == 0) {
            return;
        }
        for (int64_t peer = 0; peer < n_pes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            aclshmemx_sdma_put_nbi(gva + my_pe * MESSAGE_SIZE + data_offset, gva + my_pe * MESSAGE_SIZE + data_offset,
                                tmp_buff, ub_size, base_per_core, peer, EVENT_ID0);
        }
        aclshmemx_sdma_notify_record(tmp_buff, ub_size, EVENT_ID0);
    }
}

void test_put_notify_wait(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
{
    PutNotifyWaitTest<<<block_dim, nullptr, stream>>>(gva, config);
}

extern "C" __global__ __aicore__ void GetNotifyWaitTest(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;

    if ASCEND_IS_AIV {
        int64_t my_pe = aclshmem_my_pe();
        int64_t n_pes = aclshmem_n_pes();

        // Define temporary UB buffer for SDMA operations
        constexpr uint32_t ub_offset = 1024;
        constexpr uint32_t ub_size = 64;  // 64B for temporary buffer
        __ubuf__ uint8_t *tmp_buff = reinterpret_cast<__ubuf__ uint8_t *>(uint64_t(ub_offset));

        uint32_t data_length = static_cast<uint32_t>(MESSAGE_SIZE);
        const auto cur_block_idx = AscendC::GetBlockIdx();
        const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint64_t base_per_core = data_length / comm_block_dim;
        uint64_t extra_bytes = data_length % comm_block_dim;
        uint64_t data_offset = 0;
        if (cur_block_idx < extra_bytes) {
            data_offset = cur_block_idx * (base_per_core + 1);
        } else {
            data_offset = extra_bytes * (base_per_core + 1) + (cur_block_idx - extra_bytes) * base_per_core;
        }
        if (cur_block_idx < extra_bytes) {
            base_per_core += 1;
        }
        if (base_per_core == 0) {
            return;
        }
        for (int64_t peer = 0; peer < n_pes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            aclshmemx_sdma_get_nbi(gva + peer * MESSAGE_SIZE + data_offset, gva + peer * MESSAGE_SIZE + data_offset,
                                tmp_buff, ub_size, base_per_core, peer, EVENT_ID0);
        }
        aclshmemx_sdma_notify_record(tmp_buff, ub_size, EVENT_ID0);
    }
}

void test_get_notify_wait(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
{
    GetNotifyWaitTest<<<block_dim, nullptr, stream>>>(gva, config);
}

extern "C" __global__ __aicore__ void PutNotifyWaitTestTensor(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;

    if ASCEND_IS_AIV {
        int64_t my_pe = aclshmem_my_pe();
        int64_t n_pes = aclshmem_n_pes();

        // Define temporary UB buffer as LocalTensor for SDMA operations
        constexpr uint32_t ub_offset = 1024;
        constexpr uint32_t ub_size = 64;  // 64B for temporary buffer
        AscendC::LocalTensor<uint8_t> tmp_local;
        tmp_local.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        tmp_local.address_.bufferAddr = ub_offset;
        tmp_local.address_.dataLen = ub_size;

        constexpr uint64_t elem_size = MESSAGE_SIZE;
        const auto cur_block_idx = AscendC::GetBlockIdx();
        const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint64_t base_per_core = elem_size / comm_block_dim;
        uint64_t extra_size = elem_size % comm_block_dim;
        uint64_t data_offset = 0;
        if (cur_block_idx < extra_size) {
            data_offset = cur_block_idx * (base_per_core + 1);
        } else {
            data_offset = extra_size * (base_per_core + 1) + (cur_block_idx - extra_size) * base_per_core;
        }
        if (cur_block_idx < extra_size) {
            base_per_core += 1;
        }
        if (base_per_core == 0) {
            return;
        }
        GM_ADDR data_addr = gva + my_pe * MESSAGE_SIZE + data_offset;
        AscendC::GlobalTensor<uint8_t> src_tensor;
        src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(data_addr), base_per_core);
        AscendC::GlobalTensor<uint8_t> dst_tensor;
        dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(data_addr), base_per_core);
        for (int64_t peer = 0; peer < n_pes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            aclshmemx_sdma_put_nbi(dst_tensor, src_tensor, tmp_local, base_per_core, peer, EVENT_ID0);
        }
        aclshmemx_sdma_notify_record(tmp_local, EVENT_ID0);
    }
}

void test_put_tensor_notify_wait(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
{
    PutNotifyWaitTestTensor<<<block_dim, nullptr, stream>>>(gva, config);
}

extern "C" __global__ __aicore__ void GetNotifyWaitTestTensor(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;

    if ASCEND_IS_AIV {
        int64_t my_pe = aclshmem_my_pe();
        int64_t n_pes = aclshmem_n_pes();

        // Define temporary UB buffer as LocalTensor for SDMA operations
        constexpr uint32_t ub_offset = 1024;
        constexpr uint32_t ub_size = 64;  // 64B for temporary buffer
        AscendC::LocalTensor<uint8_t> tmp_local;
        tmp_local.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        tmp_local.address_.bufferAddr = ub_offset;
        tmp_local.address_.dataLen = ub_size;

        constexpr uint64_t elem_size = MESSAGE_SIZE;
        const auto cur_block_idx = AscendC::GetBlockIdx();
        const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint64_t base_per_core = elem_size / comm_block_dim;
        uint64_t extra_size = elem_size % comm_block_dim;
        uint64_t data_offset = 0;
        if (cur_block_idx < extra_size) {
            data_offset = cur_block_idx * (base_per_core + 1);
        } else {
            data_offset = extra_size * (base_per_core + 1) + (cur_block_idx - extra_size) * base_per_core;
        }
        if (cur_block_idx < extra_size) {
            base_per_core += 1;
        }
        if (base_per_core == 0) {
            return;
        }
        GM_ADDR data_addr = gva + my_pe * MESSAGE_SIZE + data_offset;
        AscendC::GlobalTensor<uint8_t> src_tensor;
        src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(data_addr), base_per_core);
        AscendC::GlobalTensor<uint8_t> dst_tensor;
        dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(data_addr), base_per_core);
        for (int64_t peer = 0; peer < n_pes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            aclshmemx_sdma_put_nbi(dst_tensor, src_tensor, tmp_local, base_per_core, peer, EVENT_ID0);
        }
        aclshmemx_sdma_notify_record(tmp_local, EVENT_ID0);
    }
}

void test_get_tensor_notify_wait(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)
{
    GetNotifyWaitTestTensor<<<block_dim, nullptr, stream>>>(gva, config);
}

__global__ __aicore__ void device_copy(GM_ADDR src, GM_ADDR dst, int message_length)
{
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();

    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    int64_t my_pe = aclshmem_my_pe();
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.sync_id;
    aclshmemx_mte_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub),
                          copy_ub_size, message_length, my_pe, copy_event_id);
    aclshmem_quiet();
}

void copy_demo(uint32_t block_dim, void* stream, uint8_t* src, uint8_t* dst, int elements)
{
    device_copy<<<block_dim, nullptr, stream>>>(src, dst, elements);
}
