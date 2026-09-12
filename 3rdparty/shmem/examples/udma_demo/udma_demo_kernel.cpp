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

constexpr uint64_t INIT_DUMP_SIZE = 200 * 1024 * 1024;

// Minimal all-gather implementation using UDMA.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void udma_all_gather_kernel(
    GM_ADDR gva, GM_ADDR dump, int message_length)
{
    AscendC::TPipe pipe;
#if ASCENDC_DUMP == 1
    AscendC::InitDump(false, dump, INIT_DUMP_SIZE);
#endif
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    // A user-provided LocalTensor of at least 64 bytes is required to submit UDMA tasks.
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

    int64_t my_pe = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();
    // Push the local segment to every other PE.
    for (int i = 0; i < pe_size; i++) {
        if (i == my_pe) {
            continue;
        }
        aclshmemx_udma_put_nbi(
            gva + message_length * my_pe, gva + message_length * my_pe, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(),
            message_length, i);
        aclshmemx_udma_quiet(i);
    }
    aclshmemx_barrier_all_vec();
}

void launch_udma_all_gather(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dump, int elements)
{
    udma_all_gather_kernel<<<block_dim, nullptr, stream>>>(gva, dump, elements);
}

// UDMA put with signal example
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void udma_put_signal_kernel(
    GM_ADDR gva, GM_ADDR sig_addr, GM_ADDR dump_addr, int message_length, uint64_t signal)
{
#if ASCENDC_DUMP == 1
    AscendC::InitDump(false, dump_addr, INIT_DUMP_SIZE);
#endif

    int64_t my_pe = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    // Push the local segment to every other PE with signal
    for (int i = 0; i < pe_size; i++) {
        if (i == my_pe) {
            continue;
        }
        // Use different signal address for each PE to avoid overwriting
        auto dst_sig_addr = sig_addr + sizeof(uint64_t) * my_pe;
        aclshmemx_udma_put_signal_nbi(
            gva + message_length * my_pe, gva + message_length * my_pe, message_length, (__gm__ uint64_t*)dst_sig_addr,
            signal, i);
        aclshmemx_udma_quiet(i);
    }
    aclshmemx_barrier_all_vec();
}

void launch_udma_put_signal(
    uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr, uint8_t* dump_addr, int elements,
    uint64_t signal)
{
    udma_put_signal_kernel<<<block_dim, nullptr, stream>>>(gva, sig_addr, dump_addr, elements, signal);
}