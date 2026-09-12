/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccs_sio_link_kernel.h"
#include "kernel_operator.h"
#include "shmem.h"

template <typename T>
[[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_get_kernel(GM_ADDR dst_gva, GM_ADDR src_gva,
                                           int elements, int peer_pe, int ub_size_kb)
{
    __gm__ T *dst_gm = (__gm__ T *)dst_gva;
    __gm__ T *src_gm = (__gm__ T *)src_gva;

    int32_t block_elements = elements / AscendC::GetBlockNum();
    int32_t current_block_index = AscendC::GetBlockIdx();
    int32_t offset = block_elements * current_block_index;
    if (current_block_index == AscendC::GetBlockNum() - 1) {
        block_elements = elements - offset;
    }
    if (block_elements <= 0) return;
    int ub_size_bytes = ub_size_kb * 1024;
    constexpr uint32_t sync_id = 0;

    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(sync_id);

    aclshmemx_mte_get_nbi(dst_gm + offset, src_gm + offset,
                          reinterpret_cast<__ubuf__ T *>(0), ub_size_bytes,
                          block_elements, peer_pe, sync_id);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
}

template <typename T>
__aicore__ inline void local_copy_gm2ub(__ubuf__ T *dst_ub, __gm__ T *src_gm, uint32_t size)
{
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> gm_tensor;
    AscendC::DataCopyExtParams params(1, size, 0, 0, 0);
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dst_ub);
    ub_tensor.address_.dataLen = ALIGN_UP(size, UB_ALIGN_SIZE);
    gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src_gm));
    AscendC::DataCopyPadExtParams<T> pad_params;
    AscendC::DataCopyPad(ub_tensor, gm_tensor, params, pad_params);
}

template <typename T>
__aicore__ inline void local_copy_ub2gm(__gm__ T *dst_gm, __ubuf__ T *src_ub, uint32_t size)
{
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> gm_tensor;
    AscendC::DataCopyExtParams params(1, size, 0, 0, 0);
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(src_ub);
    ub_tensor.address_.dataLen = ALIGN_UP(size, UB_ALIGN_SIZE);
    gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst_gm));
    AscendC::DataCopyPad(gm_tensor, ub_tensor, params);
}



template <typename T>
__aicore__ inline void local_mte_get_hccs(__gm__ T *hccs_src, __gm__ T *local_dst,
                                           __ubuf__ T *buf, uint32_t ub_size_bytes,
                                           uint32_t elem_size, uint32_t sync_id)
{
    uint64_t block_size = ub_size_bytes / sizeof(T) * sizeof(T);
    if (block_size == 0) {
        return;
    }
    uint64_t elem_bytes = static_cast<uint64_t>(elem_size) * sizeof(T);
    uint64_t remain = elem_bytes % block_size;
    uint64_t repeat_times = elem_bytes / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;

    for (uint64_t i = 0; i < repeat_times; i++) {
        local_copy_gm2ub(buf, hccs_src + i * repeat_elem, block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        local_copy_ub2gm(local_dst + i * repeat_elem, buf, block_size);
        if (i != loop_times - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(sync_id);
        }
    }
    if (remain > 0) {
        local_copy_gm2ub(buf, hccs_src + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(sync_id);
        local_copy_ub2gm(local_dst + repeat_times * repeat_elem, buf, remain);
    }
}

template <typename T>
[[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_get_kernel_hccs(GM_ADDR local_buf_gva, GM_ADDR hccs_src_gva,
                                                int elements, int ub_size_kb)
{
    __gm__ T *local_buf = (__gm__ T *)local_buf_gva;
    __gm__ T *hccs_src = (__gm__ T *)hccs_src_gva;

    int32_t block_elements = elements / AscendC::GetBlockNum();
    int32_t current_block_index = AscendC::GetBlockIdx();
    int32_t offset = block_elements * current_block_index;
    if (current_block_index == AscendC::GetBlockNum() - 1) {
        block_elements = elements - offset;
    }
    if (block_elements <= 0) return;
    int ub_size_bytes = ub_size_kb * 1024;
    constexpr uint32_t sync_id = 0;

    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(sync_id);

    local_mte_get_hccs(hccs_src + offset, local_buf + offset,
                       reinterpret_cast<__ubuf__ T *>(0), ub_size_bytes,
                       block_elements, sync_id);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
}

template <typename T>
[[bisheng::core_ratio(0,1)]] __global__ __aicore__ void mte_get_mixed_kernel(GM_ADDR sio_dst_gva, GM_ADDR sio_src_gva,
                                                 int sio_elements, int peer_pe,
                                                 GM_ADDR hccs_dst_gva,
                                                 GM_ADDR hccs_src_gva, int hccs_elements,
                                                 int ub_size_kb)
{
    int32_t current_block_index = AscendC::GetBlockIdx();
    int32_t total_blocks = AscendC::GetBlockNum();
    int ub_size_bytes = ub_size_kb * 1024;

    int32_t sio_block_count = total_blocks * 3 / 5;
    if (sio_block_count == 0) sio_block_count = 1;
    if (sio_block_count > total_blocks) sio_block_count = total_blocks;
    int32_t hccs_block_count = total_blocks - sio_block_count;

    constexpr uint32_t sync_id = 0;

    if (current_block_index < sio_block_count && sio_elements > 0) {
        __gm__ T *sio_dst = (__gm__ T *)sio_dst_gva;
        __gm__ T *sio_src = (__gm__ T *)sio_src_gva;

        int32_t sio_block_idx = current_block_index;
        int32_t sio_offset_elem = sio_elements / sio_block_count;
        int32_t sio_offset = sio_offset_elem * sio_block_idx;
        int32_t sio_block_elements = (sio_block_idx == sio_block_count - 1)
                                         ? sio_elements - sio_offset
                                         : sio_offset_elem;

        if (sio_block_elements > 0) {
            __ubuf__ T *ub_buf = reinterpret_cast<__ubuf__ T *>(0);

            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(sync_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(sync_id);

            aclshmemx_mte_get_nbi(sio_dst + sio_offset, sio_src + sio_offset, ub_buf, ub_size_bytes,
                                  sio_block_elements, peer_pe, sync_id);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
        }
    } else if (hccs_block_count > 0 && hccs_elements > 0) {
        __gm__ T *hccs_dst = (__gm__ T *)hccs_dst_gva;
        __gm__ T *hccs_src = (__gm__ T *)hccs_src_gva;

        int32_t hccs_block_idx = current_block_index - sio_block_count;
        int32_t hccs_offset_elem = hccs_elements / hccs_block_count;
        int32_t hccs_offset = hccs_offset_elem * hccs_block_idx;
        int32_t hccs_block_elements = (hccs_block_idx == hccs_block_count - 1)
                                          ? hccs_elements - hccs_offset
                                          : hccs_offset_elem;

        if (hccs_block_elements > 0) {
            __ubuf__ T *ub_buf = reinterpret_cast<__ubuf__ T *>(0);

            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(sync_id);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(sync_id);

            local_mte_get_hccs(hccs_src + hccs_offset, hccs_dst + hccs_offset, ub_buf, ub_size_bytes,
                               hccs_block_elements, sync_id);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(sync_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(sync_id);
        }
    }
}

template <typename T>
void launch_mte_get(uint32_t block_dim, void *stream, uint8_t *local_buf, uint8_t *src_addr,
                    int elements, int peer_pe, int ub_size_kb)
{
    mte_get_kernel<T><<<block_dim, nullptr, stream>>>(
        local_buf, src_addr, elements, peer_pe, ub_size_kb);
}

template <typename T>
void launch_mte_get_hccs(uint32_t block_dim, void *stream, uint8_t *local_buf, uint8_t *hccs_src,
                          int elements, int ub_size_kb)
{
    mte_get_kernel_hccs<T><<<block_dim, nullptr, stream>>>(
        local_buf, hccs_src, elements, ub_size_kb);
}

template <typename T>
void launch_mte_get_mixed(uint32_t block_dim, void *stream,
                           uint8_t *sio_dst, uint8_t *sio_src, int sio_elements, int peer_pe,
                           uint8_t *hccs_dst, uint8_t *hccs_src, int hccs_elements,
                           int ub_size_kb)
{
    mte_get_mixed_kernel<T><<<block_dim, nullptr, stream>>>(
        sio_dst, sio_src, sio_elements, peer_pe,
        hccs_dst, hccs_src, hccs_elements,
        ub_size_kb);
}

template void launch_mte_get<int>(uint32_t, void *, uint8_t *, uint8_t *, int, int, int);
template void launch_mte_get<int64_t>(uint32_t, void *, uint8_t *, uint8_t *, int, int, int);
template void launch_mte_get<float>(uint32_t, void *, uint8_t *, uint8_t *, int, int, int);

template void launch_mte_get_hccs<int>(uint32_t, void *, uint8_t *, uint8_t *, int, int);
template void launch_mte_get_hccs<int64_t>(uint32_t, void *, uint8_t *, uint8_t *, int, int);
template void launch_mte_get_hccs<float>(uint32_t, void *, uint8_t *, uint8_t *, int, int);

template void launch_mte_get_mixed<int>(uint32_t, void *, uint8_t *, uint8_t *, int, int, uint8_t *, uint8_t *, int, int);
template void launch_mte_get_mixed<int64_t>(uint32_t, void *, uint8_t *, uint8_t *, int, int, uint8_t *, uint8_t *, int, int);
template void launch_mte_get_mixed<float>(uint32_t, void *, uint8_t *, uint8_t *, int, int, uint8_t *, uint8_t *, int, int);
