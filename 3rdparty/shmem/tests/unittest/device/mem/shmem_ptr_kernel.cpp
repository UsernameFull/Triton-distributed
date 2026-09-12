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

class kernel_ptr_test {
public:
    __aicore__ inline kernel_ptr_test() {}
    __aicore__ inline void Init(GM_ADDR gva)
    {
        gva_gm = (__gm__ int *)gva;
        rank = aclshmem_my_pe();
        rank_size = aclshmem_n_pes();
    }
    __aicore__ inline void Process()
    {
        AscendC::PipeBarrier<PIPE_ALL>();

        __gm__ int *p0 = static_cast<__gm__ int*>(aclshmem_ptr(gva_gm, rank));
        __gm__ int *p1 = static_cast<__gm__ int*>(aclshmem_ptr(gva_gm + 1, rank));
        int32_t delta_self = p1 - p0;
        aclshmem_int32_p(gva_gm, delta_self, rank);

        int peer = (rank + 1) % rank_size;
        __gm__ int *q0 = static_cast<__gm__ int*>(aclshmem_ptr(gva_gm, peer));
        __gm__ int *q1 = static_cast<__gm__ int*>(aclshmem_ptr(gva_gm + 1, peer));
        int32_t delta_remote = q1 - q0;
        aclshmem_int32_p(gva_gm + 1, delta_remote, rank);
    }

private:
    __gm__ int *gva_gm;
    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void device_ptr_test(GM_ADDR gva)
{
    kernel_ptr_test op;
    op.Init(gva);
    op.Process();
}

void get_device_ptr(uint32_t block_dim, void* stream, uint8_t* gva)
{
    device_ptr_test<<<block_dim, nullptr, stream>>>(gva);
}
