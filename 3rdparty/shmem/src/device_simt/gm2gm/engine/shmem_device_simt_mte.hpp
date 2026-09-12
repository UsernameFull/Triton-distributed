/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_SIMT_GM2GM_MTE_HPP
#define SHMEM_DEVICE_SIMT_GM2GM_MTE_HPP

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "ub2gm/mte/shmemi_device_mte.h"
#include "shmemi_device_simt_common.h"

namespace simt {

__simt_callee__ inline bool is_host_mem_heap(__gm__ void *ptr)
{
    __gm__ aclshmem_device_host_state_simt_t *device_state = simt::aclshmemi_get_state();
    if ((__gm__ void *)device_state->host_heap_base == nullptr) {
        return false;
    }
    uint64_t ptr_uint = reinterpret_cast<uint64_t>(ptr);
    uint64_t base_uint = reinterpret_cast<uint64_t>(device_state->host_heap_base);
    uint64_t end_uint = base_uint + device_state->heap_size;
    return (ptr_uint >= base_uint && ptr_uint < end_uint);
}

__simt_callee__ inline bool assert_remote_ptr_valid(__gm__ void *ptr, uint64_t remote_ptr, int pe)
{
    if (!is_host_mem_heap(ptr)) {
        return true;
    }
    __gm__ aclshmem_device_host_state_simt_t *device_state = aclshmemi_get_state();
    uint64_t host_heap_offset =
        reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(device_state->host_heap_base);
    uint64_t remote_host_ptr = reinterpret_cast<uint64_t>(((__gm__ uint64_t*)(device_state->p2p_host_heap_base))[pe]) + host_heap_offset;
    if (remote_host_ptr != remote_ptr) {
        // ACLSHMEM_DEBUG_FUNC(aclshmemi_kernel_abort, "Host mem remote ptr %ld is invalid, expect: %ld\n", remote_ptr, remote_host_ptr);
        return false;
    }
    return true;
}

__simt_callee__ inline __gm__ void *aclshmem_ptr(__gm__ void *ptr, int pe)
{
    // Get Global State
    __gm__ aclshmem_device_host_state_simt_t *device_state = simt::aclshmemi_get_state();
    // Back to root address
    ptrdiff_t offset = reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(device_state->heap_base);
    // Address translate
    uintptr_t remote_ptr = reinterpret_cast<uintptr_t>(((__gm__ uint64_t*)(device_state->p2p_device_heap_base))[pe]) + offset;
    
    {
        #if defined(ASCENDC_DEBUG) && defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 1) && defined(DEBUG_MODE)
        // ACLSHMEM_DEBUG_FUNC(assert_remote_ptr_valid, ptr, (uint64_t)remote_ptr, pe);
        if (assert_remote_ptr_valid(ptr, (uint64_t)remote_ptr, pe)) {
            return;
        }
        #endif
    }

    return reinterpret_cast<__gm__ void *>(remote_ptr);
}

template<typename T, thread_group_t scope>
__simt_callee__ inline void aclshmemi_mte_get_nbi(
    __gm__ T *dst, __gm__ T *src, size_t elem_size, int pe
) 
{
    aclshmemi_threadgroup_sync<scope>();
    size_t myIndex = aclshmemi_thread_id_in_threadgroup<scope>();
    size_t groupSize = aclshmemi_threadgroup_size<scope>();

    __gm__ T* srcSym = (__gm__ T*)aclshmem_ptr(src, pe);

    for (size_t i = myIndex; i < elem_size; i += groupSize) {
        dst[i] = srcSym[i];
    }
    aclshmemi_threadgroup_sync<scope>();
}

template <typename T, thread_group_t scope>
__simt_callee__ inline void aclshmemi_mte_put_nbi(
    __gm__ T *dst, __gm__ T *src, size_t elem_size, int pe
)
{
    aclshmemi_threadgroup_sync<scope>();

    size_t myIndex = aclshmemi_thread_id_in_threadgroup<scope>();
    size_t groupSize = aclshmemi_threadgroup_size<scope>();

    __gm__ T* dstSym = (__gm__ T*)aclshmem_ptr(dst, pe);

    for (size_t i = myIndex; i < elem_size; i += groupSize) {
        dstSym[i] = src[i];
    }
    aclshmemi_threadgroup_sync<scope>();
}

template<typename T>
__simt_callee__ inline void aclshmemx_mte_get_nbi(__gm__ T *dst, __gm__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_get_nbi<T, ACLSHMEMI_THREADGROUP_THREAD>(dst, src, elem_size, pe);
}

template<typename T>
__simt_callee__ inline void aclshmemx_mte_get_nbi_block(__gm__ T *dst, __gm__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_get_nbi<T, ACLSHMEMI_THREADGROUP_BLOCK>(dst, src, elem_size, pe);
}

template<typename T>
__simt_callee__ inline void aclshmemx_mte_get_nbi_warp(__gm__ T *dst, __gm__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_get_nbi<T, ACLSHMEMI_THREADGROUP_WARP>(dst, src, elem_size, pe);
}

template<typename T>
__simt_callee__ inline void aclshmemx_mte_put_nbi(__gm__ T *dst, __gm__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_put_nbi<T, ACLSHMEMI_THREADGROUP_THREAD>(dst, src, elem_size, pe);
}

template<typename T>
__simt_callee__ inline void aclshmemx_mte_put_nbi_block(__gm__ T *dst, __gm__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_put_nbi<T, ACLSHMEMI_THREADGROUP_BLOCK>(dst, src, elem_size, pe);
}

template<typename T>
__simt_callee__ inline void aclshmemx_mte_put_nbi_warp(__gm__ T *dst, __gm__ T *src, size_t elem_size, int pe)
{
    aclshmemi_mte_put_nbi<T, ACLSHMEMI_THREADGROUP_WARP>(dst, src, elem_size, pe);
}

} // namespace simt

#endif