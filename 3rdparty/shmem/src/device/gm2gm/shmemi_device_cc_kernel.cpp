/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem.h"
#include "host_device/shmem_common_types.h"
#include "shmemi_device_common.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"

// kernels
ACLSHMEM_GLOBAL void barrier_on_stream_kernel(aclshmem_team_t team, uint64_t ffts_config)
{
    AscendC::SetSyncBaseAddr(ffts_config);
    aclshmemi_barrier<false>(team);
}

// interfaces
 int32_t aclshmemi_call_barrier_on_stream_kernel(aclshmem_team_t team, aclrtStream stream)
{
    barrier_on_stream_kernel<<<1, nullptr, stream>>>(team, util_get_ffts_config());
    return ACLSHMEM_SUCCESS;
}

// kernels
ACLSHMEM_GLOBAL_VECTOR void k_aclshmem_handle_wait(int32_t tid)
{
    aclshmemi_handle(tid);
}

// interfaces
void aclshmemi_handle_wait_on_stream(aclshmem_handle_t handle, aclrtStream stream)
{
    // call barrier kernel
    k_aclshmem_handle_wait<<<1, nullptr, stream>>>((int32_t)handle.team_id);
}

// kernels
template <typename T>
ACLSHMEM_GLOBAL void k_memset(GM_ADDR array, int32_t len, T val, int32_t count)
{
    if (array == 0) {
        return;
    }
    auto tmp = (__gm__ T *)array;
    int32_t valid_count = count < len ? count : len;
    for (int32_t i = 0; i < valid_count; i++) {
        *tmp++ = val;
    }

    dcci_entire_cache();
}

// interfaces
int32_t aclshmemi_memset(int32_t *array, int32_t len, int32_t val, int32_t count)
{
    k_memset<int32_t><<<1, nullptr, nullptr>>>((uint8_t *)array, len, val, count);
    return aclrtSynchronizeStream(nullptr);
}

ACLSHMEM_GLOBAL void k_aclshmem_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val)
{
    aclshmem_signal_wait_until(sig_addr, cmp, cmp_val);
}

void call_aclshmemi_signal_wait_until_on_stream_kernel(int32_t *sig_addr, int cmp, int32_t cmp_val, aclrtStream stream)
{
    k_aclshmem_signal_wait_until<<<1, nullptr, stream>>>(sig_addr, cmp, cmp_val);
}

ACLSHMEM_GLOBAL_VECTOR void k_aclshmem_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
{
    aclshmemx_signal_op(sig_addr, signal, sig_op, pe);
}

void call_aclshmemi_signal_op_on_stream_kernel(int32_t *sig_addr, int32_t signal, int sig_op, int pe, aclrtStream stream)
{
    k_aclshmem_signal_op<<<1, nullptr, stream>>>(sig_addr, signal, sig_op, pe);
}