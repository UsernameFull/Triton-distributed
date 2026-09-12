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

extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__ void udma_atomic_add_kernel(
    GM_ADDR gva, GM_ADDR dump, int message_length)
{
#if ASCENDC_DUMP == 1
    AscendC::InitDump(false, dump, INIT_DUMP_SIZE);
#endif

    int32_t my_rank = aclshmem_my_pe();
    int32_t pe_size = aclshmem_n_pes();
    int32_t peer_id = (my_rank + 1) % pe_size;
    AscendC::PipeBarrier<PIPE_ALL>();
    int32_t value = 10;
    aclshmemx_udma_atomic_add((__gm__ int32_t *)gva, value, peer_id);
    aclshmemx_udma_quiet(peer_id);
}

void launch_udma_atomic_add(uint32_t block_dim, void *stream, uint8_t *gva, uint8_t *dump, int elements)
{
    udma_atomic_add_kernel<<<block_dim, nullptr, stream>>>(gva, dump, elements);
}