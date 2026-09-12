/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_SIMT_COMMON_HPP
#define SHMEMI_DEVICE_SIMT_COMMON_HPP

#include "shmemi_def.h"

#include "shmemi_device_simt_meta.h"

namespace simt
{

__simt_callee__ inline __gm__ aclshmem_device_host_state_simt_t *aclshmemi_get_state() {
    return reinterpret_cast<__gm__ aclshmem_device_host_state_simt_t *>(aclshmemi_get_extra_context_addr(0));
}

__simt_callee__ inline int aclshmemi_get_my_pe()
{
    return aclshmemi_get_state()->mype;
}

__simt_callee__ inline int aclshmemi_get_total_pe()
{
    return aclshmemi_get_state()->npes;
}

__simt_callee__ inline uint64_t aclshmemi_get_heap_size()
{
    return aclshmemi_get_state()->heap_size;
}

template<typename T>
__simt_callee__ inline void aclshmemi_store(__gm__ T *addr, T val)
{
    *((__gm__ T *)addr) = val;
}

template<typename T>
__simt_callee__ inline T aclshmemi_load(__gm__ T *cache)
{
    return *((__gm__ T *)cache);
}

typedef enum {
    ACLSHMEMI_THREADGROUP_THREAD = 0,
    aclshmemi_threadgroup_thread = 0,
    ACLSHMEMI_THREADGROUP_WARP = 1,
    aclshmemi_threadgroup_warp = 1,
    ACLSHMEMI_THREADGROUP_BLOCK = 2,
    aclshmemi_threadgroup_block = 2
} thread_group_t;

template<thread_group_t scope>
__simt_callee__ inline void aclshmemi_threadgroup_sync() {
    if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_THREAD) {
        return;
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_WARP) {
        asc_syncthreads();
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_BLOCK) {
        asc_syncthreads();
    }
}

template<thread_group_t scope>
__simt_callee__ inline size_t aclshmemi_thread_id_in_threadgroup() {
    if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_THREAD) {
        return 0;
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_WARP) {
        return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) % warpSize;
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_BLOCK) {
        return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    }
}

template<thread_group_t scope>
__simt_callee__ inline size_t aclshmemi_threadgroup_size() {
    if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_THREAD) {
        return 1;
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_WARP) {
        size_t actualWarpSize = blockDim.x * blockDim.y * blockDim.z;
        return (actualWarpSize < warpSize ? actualWarpSize : warpSize);
    } else if constexpr (scope == thread_group_t::ACLSHMEMI_THREADGROUP_BLOCK) {
        return blockDim.x * blockDim.y * blockDim.z;
    }
}

} // namespace simt

#endif