/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RDMA_DEMO_KERNEL_
#define _RDMA_DEMO_KERNEL_

#include "kernel_operator.h"
#include "shmem.h"

// all_gather简易实现
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void device_all_gather_test(GM_ADDR gva, int message_length)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    // 需要用户指定一个长度大于等于64字节的LocalTensor用于RDMA任务下发
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);
    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();
    AscendC::PipeBarrier<PIPE_ALL>();
    // All Gather
    for (int i = 0; i < pe_size; i++) {
        if (i == my_rank) {
            continue;
        }
        aclshmemx_roce_put_nbi(gva + message_length * my_rank, gva + message_length * my_rank,
                                (__ubuf__ uint8_t*)ubLocal.GetPhyAddr(), message_length, i, 0);
    }
}

void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int elements)
{
    device_all_gather_test<<<block_dim, nullptr, stream>>>(gva, elements);
}

#endif  // _RDMA_DEMO_KERNEL_