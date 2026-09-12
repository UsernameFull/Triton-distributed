/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_SIMT_GM2GM_RMA_HPP
#define SHMEM_DEVICE_SIMT_GM2GM_RMA_HPP

#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "device/gm2gm/engine/shmem_device_sdma.h"
#include "device/shmem_def.h"
#include "gm2gm/engine/shmemi_device_rdma.h"
#include "host/shmem_host_def.h"
#include "host_device/shmem_common_macros.h"
#include "simt_api/device_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace simt
{
#define ACLSHMEM_TYPE_FUNC_SIMT(FUNC) \
    FUNC(half, half);            \
    FUNC(float, float);          \
    FUNC(int8, int8_t);          \
    FUNC(int16, int16_t);        \
    FUNC(int32, int32_t);        \
    FUNC(int64, int64_t);        \
    FUNC(uint8, uint8_t);        \
    FUNC(uint16, uint16_t);      \
    FUNC(uint32, uint32_t);      \
    FUNC(uint64, uint64_t);      \
    FUNC(char, signed char);     \
    FUNC(char, unsigned char);   \
    FUNC(bfloat16, bfloat16_t)

// ========================================================

#define ACLSHMEM_PUT_TYPENAME_MEM_SIMT(NAME, TYPE) \
    __simt_callee__ inline void aclshmem_##NAME##_put(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe)   \
    {                                                                                                                       \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_THREAD>(dst, src, elem_size, pe);                                 \
    }                                                                                                                       \
    __simt_callee__ inline void aclshmemx_##NAME##_put_warp(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe) \
    {                                                                                                                       \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_WARP>(dst, src, elem_size, pe);                                   \
    }                                                                                                                       \
    __simt_callee__ inline void aclshmemx_##NAME##_put_block(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe) \
    {                                                                                                                       \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_BLOCK>(dst, src, elem_size, pe);                                  \
    }

ACLSHMEM_TYPE_FUNC_SIMT(ACLSHMEM_PUT_TYPENAME_MEM_SIMT);

// ========================================================

#define ACLSHMEM_PUT_BITS_SIMT(BITS, TYPE) \
    __simt_callee__ inline void aclshmem_put##BITS(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_THREAD>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(src), \
            nelems, \
            pe \
        ); \
    } \
    __simt_callee__ inline void aclshmemx_put##BITS##_warp(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_WARP>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(src), \
            nelems, \
            pe \
        ); \
    } \
    __simt_callee__ inline void aclshmemx_put##BITS##_block(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_BLOCK>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(src), \
            nelems, \
            pe \
        ); \
    }

ACLSHMEM_PUT_BITS_SIMT(8, int8_t)
ACLSHMEM_PUT_BITS_SIMT(16, int16_t)
ACLSHMEM_PUT_BITS_SIMT(32, int32_t)
ACLSHMEM_PUT_BITS_SIMT(64, int64_t)
ACLSHMEM_PUT_BITS_SIMT(128, int4)

// ========================================================

__simt_callee__ inline void aclshmem_putmem(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_put_nbi<char, ACLSHMEMI_THREADGROUP_THREAD>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

__simt_callee__ inline void aclshmemx_putmem_warp(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_put_nbi<char, ACLSHMEMI_THREADGROUP_WARP>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

__simt_callee__ inline void aclshmemx_putmem_block(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_put_nbi<char, ACLSHMEMI_THREADGROUP_BLOCK>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

// ========================================================

#define ACLSHMEM_TYPENAME_P_AICORE_SIMT(NAME, TYPE)                                              \
    __simt_callee__ inline void aclshmem_##NAME##_p(__gm__ TYPE *dst, const TYPE value, int pe)    \
    {                                                                                       \
        auto ptr = simt::aclshmem_ptr(dst, pe);                                             \
        __gm__ TYPE *addr_gm = reinterpret_cast<__gm__ TYPE *>(ptr);                        \
                                                                                            \
        asc_stwt(addr_gm, value);                                                           \
    }

ACLSHMEM_TYPE_FUNC_SIMT(ACLSHMEM_TYPENAME_P_AICORE_SIMT);

// ========================================================

#define ACLSHMEM_GET_TYPENAME_MEM_SIMT(NAME, TYPE)                                                                          \
    __simt_callee__ inline void aclshmem_##NAME##_get(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe)   \
    {                                                                                                                       \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_THREAD>(dst, src, elem_size, pe);                                 \
    }                                                                                                                       \
    __simt_callee__ inline void aclshmemx_##NAME##_get_warp(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe) \
    {                                                                                                                       \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_WARP>(dst, src, elem_size, pe);                                   \
    }                                                                                                                       \
    __simt_callee__ inline void aclshmemx_##NAME##_get_block(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe) \
    {                                                                                                                       \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_BLOCK>(dst, src, elem_size, pe);                                  \
    }

ACLSHMEM_TYPE_FUNC_SIMT(ACLSHMEM_GET_TYPENAME_MEM_SIMT);

// ========================================================

#define ACLSHMEM_GET_BITS_SIMT(BITS, TYPE) \
    __simt_callee__ inline void aclshmem_get##BITS(__gm__ void* dst, __gm__ void* source, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_THREAD>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(source), \
            nelems, \
            pe \
        ); \
    } \
    __simt_callee__ inline void aclshmemx_get##BITS##_warp(__gm__ void* dst, __gm__ void* source, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_WARP>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(source), \
            nelems, \
            pe \
        ); \
    } \
    __simt_callee__ inline void aclshmemx_get##BITS##_block(__gm__ void* dst, __gm__ void* source, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_BLOCK>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(source), \
            nelems, \
            pe \
        ); \
    }

ACLSHMEM_GET_BITS_SIMT(8, int8_t)
ACLSHMEM_GET_BITS_SIMT(16, int16_t)
ACLSHMEM_GET_BITS_SIMT(32, int32_t)
ACLSHMEM_GET_BITS_SIMT(64, int64_t)
ACLSHMEM_GET_BITS_SIMT(128, int4)

// ========================================================

__simt_callee__ inline void aclshmem_getmem(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_get_nbi<char, ACLSHMEMI_THREADGROUP_THREAD>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

__simt_callee__ inline void aclshmemx_getmem_warp(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_get_nbi<char, ACLSHMEMI_THREADGROUP_WARP>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

__simt_callee__ inline void aclshmemx_getmem_block(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_get_nbi<char, ACLSHMEMI_THREADGROUP_BLOCK>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

// ========================================================

#define ACLSHMEM_TYPENAME_G_AICORE_SIMT(NAME, TYPE)                                         \
    __simt_callee__ inline TYPE aclshmem_##NAME##_g(__gm__ TYPE *src, int32_t pe)           \
    {                                                                                       \
        auto ptr = simt::aclshmem_ptr(src, pe);                                             \
        __gm__ TYPE *addr_gm = reinterpret_cast<__gm__ TYPE *>(ptr);                        \
                                                                                            \
        return asc_ldca(addr_gm);                                                           \
    }

ACLSHMEM_TYPE_FUNC_SIMT(ACLSHMEM_TYPENAME_G_AICORE_SIMT);

// ========================================================

#define ACLSHMEM_PUT_TYPENAME_MEM_NBI_SIMT(NAME, TYPE) \
    __simt_callee__ inline void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe)   \
    {                                                                                                                           \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_THREAD>(dst, src, elem_size, pe);                                     \
    }                                                                                                                           \
    __simt_callee__ inline void aclshmemx_##NAME##_put_nbi_warp(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe) \
    {                                                                                                                           \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_WARP>(dst, src, elem_size, pe);                                       \
    }                                                                                                                           \
    __simt_callee__ inline void aclshmemx_##NAME##_put_nbi_block(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe) \
    {                                                                                                                           \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_BLOCK>(dst, src, elem_size, pe);                                      \
    }

ACLSHMEM_TYPE_FUNC_SIMT(ACLSHMEM_PUT_TYPENAME_MEM_NBI_SIMT);

// ========================================================

#define ACLSHMEM_PUT_BITS_NBI_SIMT(BITS, TYPE) \
    __simt_callee__ inline void aclshmem_put##BITS##_nbi(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_THREAD>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(src), \
            nelems, \
            pe \
        ); \
    } \
    __simt_callee__ inline void aclshmemx_put##BITS##_nbi_warp(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_WARP>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(src), \
            nelems, \
            pe \
        ); \
    } \
    __simt_callee__ inline void aclshmemx_put##BITS##_nbi_block(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_put_nbi<TYPE, ACLSHMEMI_THREADGROUP_BLOCK>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(src), \
            nelems, \
            pe \
        ); \
    }

ACLSHMEM_PUT_BITS_NBI_SIMT(8, int8_t);
ACLSHMEM_PUT_BITS_NBI_SIMT(16, int16_t);
ACLSHMEM_PUT_BITS_NBI_SIMT(32, int32_t);
ACLSHMEM_PUT_BITS_NBI_SIMT(64, int64_t);
ACLSHMEM_PUT_BITS_NBI_SIMT(128, int4);

// ========================================================

__simt_callee__ inline void aclshmem_putmem_nbi(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_put_nbi<char, ACLSHMEMI_THREADGROUP_THREAD>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

__simt_callee__ inline void aclshmemx_putmem_nbi_warp(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_put_nbi<char, ACLSHMEMI_THREADGROUP_WARP>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

__simt_callee__ inline void aclshmemx_putmem_nbi_block(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_put_nbi<char, ACLSHMEMI_THREADGROUP_BLOCK>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

// ========================================================

#define ACLSHMEM_GET_TYPENAME_MEM_NBI_SIMT(NAME, TYPE) \
    __simt_callee__ inline void aclshmem_##NAME##_get_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe)   \
    {                                                                                                                           \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_THREAD>(dst, src, elem_size, pe);                                     \
    }                                                                                                                           \
    __simt_callee__ inline void aclshmemx_##NAME##_get_nbi_warp(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe) \
    {                                                                                                                           \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_WARP>(dst, src, elem_size, pe);                                       \
    }                                                                                                                           \
    __simt_callee__ inline void aclshmemx_##NAME##_get_nbi_block(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size, int32_t pe) \
    {                                                                                                                           \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_BLOCK>(dst, src, elem_size, pe);                                      \
    }

ACLSHMEM_TYPE_FUNC_SIMT(ACLSHMEM_GET_TYPENAME_MEM_NBI_SIMT);

// ========================================================

#define ACLSHMEM_GET_BITS_NBI_SIMT(BITS, TYPE) \
    __simt_callee__ inline void aclshmem_get##BITS##_nbi(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_THREAD>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(src), \
            nelems, \
            pe \
        ); \
    } \
    __simt_callee__ inline void aclshmemx_get##BITS##_nbi_warp(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_WARP>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(src), \
            nelems, \
            pe \
        ); \
    } \
    __simt_callee__ inline void aclshmemx_get##BITS##_nbi_block(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe) \
    { \
        aclshmemi_mte_get_nbi<TYPE, ACLSHMEMI_THREADGROUP_BLOCK>( \
            reinterpret_cast<__gm__ TYPE *>(dst), \
            reinterpret_cast<__gm__ TYPE *>(src), \
            nelems, \
            pe \
        ); \
    }

ACLSHMEM_GET_BITS_NBI_SIMT(8, int8_t);
ACLSHMEM_GET_BITS_NBI_SIMT(16, int16_t);
ACLSHMEM_GET_BITS_NBI_SIMT(32, int32_t);
ACLSHMEM_GET_BITS_NBI_SIMT(64, int64_t);
ACLSHMEM_GET_BITS_NBI_SIMT(128, int4);

// ========================================================

__simt_callee__ inline void aclshmem_getmem_nbi(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_get_nbi<char, ACLSHMEMI_THREADGROUP_THREAD>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

__simt_callee__ inline void aclshmemx_getmem_nbi_warp(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_get_nbi<char, ACLSHMEMI_THREADGROUP_WARP>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

__simt_callee__ inline void aclshmemx_getmem_nbi_block(__gm__ void *dst, __gm__ void *src, size_t elem_size, int32_t pe)
{
    aclshmemi_mte_get_nbi<char, ACLSHMEMI_THREADGROUP_BLOCK>(
        reinterpret_cast<__gm__ char *>(dst),
        reinterpret_cast<__gm__ char *>(src),
        elem_size,
        pe
    );
}

} // namespace simt
#endif
