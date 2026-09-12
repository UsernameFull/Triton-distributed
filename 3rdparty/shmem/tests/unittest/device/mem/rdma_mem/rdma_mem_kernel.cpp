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

#include "shmem.h"
constexpr uint64_t MESSAGE_SIZE = 64;

extern "C" __global__ __aicore__ void RDMAGetTestLowLevel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR dest_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        dest_addr = gva + peer * MESSAGE_SIZE;
        aclshmemx_roce_get_nbi(dest_addr, dest_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), MESSAGE_SIZE, peer, 0);
    }
}

void test_rdma_get_low_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAGetTestLowLevel<<<block_dim, nullptr, stream>>>(gva, config);
}

extern "C" __global__ __aicore__ void RDMAPutTestLowLevel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR src_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        src_addr = gva + rank * MESSAGE_SIZE;
        aclshmemx_roce_put_nbi(src_addr, src_addr, (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), MESSAGE_SIZE, peer, 0);
    }
}

void test_rdma_put_low_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAPutTestLowLevel<<<block_dim, nullptr, stream>>>(gva, config);
}

extern "C" __global__ __aicore__ void RDMAGetTestHighLevel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR dest_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        dest_addr = gva + peer * MESSAGE_SIZE;
        aclshmem_uint8_get_nbi(dest_addr, dest_addr, MESSAGE_SIZE, peer);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

void test_rdma_get_high_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAGetTestHighLevel<<<block_dim, nullptr, stream>>>(gva, config);
}

extern "C" __global__ __aicore__ void RDMAPutTestHighLevel(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR src_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        src_addr = gva + rank * MESSAGE_SIZE;
        aclshmem_uint8_put_nbi(src_addr, src_addr, MESSAGE_SIZE, peer);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

void test_rdma_put_high_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAPutTestHighLevel<<<block_dim, nullptr, stream>>>(gva, config);
}