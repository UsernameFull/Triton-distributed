/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*
 * @file simt_rma_gm2gm_kernel.cpp
 * @brief Device implementation of SIMT RMA gm2gm unit tests.
 *
 * Wraps invocations of SIMT RMA gm2gm functions.
 * Each invocation process involves three functions:
 *   · void test_##FUNCNAME
 *   · __global__ __vector__ void kernel_call_##FUNCNAME
 *   · __simt_vf__ __launch_bounds__(1024) inline void simt_call_##FUNCNAME
 * These functions simulate the execution flow: Host -> aicore -> simt vf function.
 * All functions, except _g and _p variants, share the same signature: (__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe).
 * The file is structured into three sections: general functions, _p functions, and _g functions.
 * Additionally, since signed char and unsigned char share the same function name, they are handled separately.
 */

#if defined(USE_SIMT)

#include "kernel_operator.h"
#include "shmem.h"
#include "utils/debug/asc_printf.h"

// ===================general functions==================================

// Warning: DO NOT REMOVE the printf function invocation, such as printf("Testing " #FUNCNAME "\n");
// Doing so may cause mysterious compiler errors.

#define SIMT_RMA_TEST(FUNC) \
    FUNC(aclshmem_half_put, half) \
    FUNC(aclshmemx_half_put_warp, half) \
    FUNC(aclshmemx_half_put_block, half) \
    FUNC(aclshmem_half_put_nbi, half) \
    FUNC(aclshmemx_half_put_nbi_warp, half) \
    FUNC(aclshmemx_half_put_nbi_block, half) \
    FUNC(aclshmem_float_put, float) \
    FUNC(aclshmemx_float_put_warp, float) \
    FUNC(aclshmemx_float_put_block, float) \
    FUNC(aclshmem_float_put_nbi, float) \
    FUNC(aclshmemx_float_put_nbi_warp, float) \
    FUNC(aclshmemx_float_put_nbi_block, float) \
    FUNC(aclshmem_int8_put, int8_t) \
    FUNC(aclshmemx_int8_put_warp, int8_t) \
    FUNC(aclshmemx_int8_put_block, int8_t) \
    FUNC(aclshmem_int8_put_nbi, int8_t) \
    FUNC(aclshmemx_int8_put_nbi_warp, int8_t) \
    FUNC(aclshmemx_int8_put_nbi_block, int8_t) \
    FUNC(aclshmem_int16_put, int16_t) \
    FUNC(aclshmemx_int16_put_warp, int16_t) \
    FUNC(aclshmemx_int16_put_block, int16_t) \
    FUNC(aclshmem_int16_put_nbi, int16_t) \
    FUNC(aclshmemx_int16_put_nbi_warp, int16_t) \
    FUNC(aclshmemx_int16_put_nbi_block, int16_t) \
    FUNC(aclshmem_int32_put, int32_t) \
    FUNC(aclshmemx_int32_put_warp, int32_t) \
    FUNC(aclshmemx_int32_put_block, int32_t) \
    FUNC(aclshmem_int32_put_nbi, int32_t) \
    FUNC(aclshmemx_int32_put_nbi_warp, int32_t) \
    FUNC(aclshmemx_int32_put_nbi_block, int32_t) \
    FUNC(aclshmem_int64_put, int64_t) \
    FUNC(aclshmemx_int64_put_warp, int64_t) \
    FUNC(aclshmemx_int64_put_block, int64_t) \
    FUNC(aclshmem_int64_put_nbi, int64_t) \
    FUNC(aclshmemx_int64_put_nbi_warp, int64_t) \
    FUNC(aclshmemx_int64_put_nbi_block, int64_t) \
    FUNC(aclshmem_uint8_put, uint8_t) \
    FUNC(aclshmemx_uint8_put_warp, uint8_t) \
    FUNC(aclshmemx_uint8_put_block, uint8_t) \
    FUNC(aclshmem_uint8_put_nbi, uint8_t) \
    FUNC(aclshmemx_uint8_put_nbi_warp, uint8_t) \
    FUNC(aclshmemx_uint8_put_nbi_block, uint8_t) \
    FUNC(aclshmem_uint16_put, uint16_t) \
    FUNC(aclshmemx_uint16_put_warp, uint16_t) \
    FUNC(aclshmemx_uint16_put_block, uint16_t) \
    FUNC(aclshmem_uint16_put_nbi, uint16_t) \
    FUNC(aclshmemx_uint16_put_nbi_warp, uint16_t) \
    FUNC(aclshmemx_uint16_put_nbi_block, uint16_t) \
    FUNC(aclshmem_uint32_put, uint32_t) \
    FUNC(aclshmemx_uint32_put_warp, uint32_t) \
    FUNC(aclshmemx_uint32_put_block, uint32_t) \
    FUNC(aclshmem_uint32_put_nbi, uint32_t) \
    FUNC(aclshmemx_uint32_put_nbi_warp, uint32_t) \
    FUNC(aclshmemx_uint32_put_nbi_block, uint32_t) \
    FUNC(aclshmem_uint64_put, uint64_t) \
    FUNC(aclshmemx_uint64_put_warp, uint64_t) \
    FUNC(aclshmemx_uint64_put_block, uint64_t) \
    FUNC(aclshmem_uint64_put_nbi, uint64_t) \
    FUNC(aclshmemx_uint64_put_nbi_warp, uint64_t) \
    FUNC(aclshmemx_uint64_put_nbi_block, uint64_t) \
    FUNC(aclshmem_char_put, unsigned char) \
    FUNC(aclshmemx_char_put_warp, unsigned char) \
    FUNC(aclshmemx_char_put_block, unsigned char) \
    FUNC(aclshmem_char_put_nbi, unsigned char) \
    FUNC(aclshmemx_char_put_nbi_warp, unsigned char) \
    FUNC(aclshmemx_char_put_nbi_block, unsigned char) \
    FUNC(aclshmem_bfloat16_put, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_put_warp, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_put_block, bfloat16_t) \
    FUNC(aclshmem_bfloat16_put_nbi, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_put_nbi_warp, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_put_nbi_block, bfloat16_t) \
    FUNC(aclshmem_half_get, half) \
    FUNC(aclshmemx_half_get_warp, half) \
    FUNC(aclshmemx_half_get_block, half) \
    FUNC(aclshmem_half_get_nbi, half) \
    FUNC(aclshmemx_half_get_nbi_warp, half) \
    FUNC(aclshmemx_half_get_nbi_block, half) \
    FUNC(aclshmem_float_get, float) \
    FUNC(aclshmemx_float_get_warp, float) \
    FUNC(aclshmemx_float_get_block, float) \
    FUNC(aclshmem_float_get_nbi, float) \
    FUNC(aclshmemx_float_get_nbi_warp, float) \
    FUNC(aclshmemx_float_get_nbi_block, float) \
    FUNC(aclshmem_int8_get, int8_t) \
    FUNC(aclshmemx_int8_get_warp, int8_t) \
    FUNC(aclshmemx_int8_get_block, int8_t) \
    FUNC(aclshmem_int8_get_nbi, int8_t) \
    FUNC(aclshmemx_int8_get_nbi_warp, int8_t) \
    FUNC(aclshmemx_int8_get_nbi_block, int8_t) \
    FUNC(aclshmem_int16_get, int16_t) \
    FUNC(aclshmemx_int16_get_warp, int16_t) \
    FUNC(aclshmemx_int16_get_block, int16_t) \
    FUNC(aclshmem_int16_get_nbi, int16_t) \
    FUNC(aclshmemx_int16_get_nbi_warp, int16_t) \
    FUNC(aclshmemx_int16_get_nbi_block, int16_t) \
    FUNC(aclshmem_int32_get, int32_t) \
    FUNC(aclshmemx_int32_get_warp, int32_t) \
    FUNC(aclshmemx_int32_get_block, int32_t) \
    FUNC(aclshmem_int32_get_nbi, int32_t) \
    FUNC(aclshmemx_int32_get_nbi_warp, int32_t) \
    FUNC(aclshmemx_int32_get_nbi_block, int32_t) \
    FUNC(aclshmem_int64_get, int64_t) \
    FUNC(aclshmemx_int64_get_warp, int64_t) \
    FUNC(aclshmemx_int64_get_block, int64_t) \
    FUNC(aclshmem_int64_get_nbi, int64_t) \
    FUNC(aclshmemx_int64_get_nbi_warp, int64_t) \
    FUNC(aclshmemx_int64_get_nbi_block, int64_t) \
    FUNC(aclshmem_uint8_get, uint8_t) \
    FUNC(aclshmemx_uint8_get_warp, uint8_t) \
    FUNC(aclshmemx_uint8_get_block, uint8_t) \
    FUNC(aclshmem_uint8_get_nbi, uint8_t) \
    FUNC(aclshmemx_uint8_get_nbi_warp, uint8_t) \
    FUNC(aclshmemx_uint8_get_nbi_block, uint8_t) \
    FUNC(aclshmem_uint16_get, uint16_t) \
    FUNC(aclshmemx_uint16_get_warp, uint16_t) \
    FUNC(aclshmemx_uint16_get_block, uint16_t) \
    FUNC(aclshmem_uint16_get_nbi, uint16_t) \
    FUNC(aclshmemx_uint16_get_nbi_warp, uint16_t) \
    FUNC(aclshmemx_uint16_get_nbi_block, uint16_t) \
    FUNC(aclshmem_uint32_get, uint32_t) \
    FUNC(aclshmemx_uint32_get_warp, uint32_t) \
    FUNC(aclshmemx_uint32_get_block, uint32_t) \
    FUNC(aclshmem_uint32_get_nbi, uint32_t) \
    FUNC(aclshmemx_uint32_get_nbi_warp, uint32_t) \
    FUNC(aclshmemx_uint32_get_nbi_block, uint32_t) \
    FUNC(aclshmem_uint64_get, uint64_t) \
    FUNC(aclshmemx_uint64_get_warp, uint64_t) \
    FUNC(aclshmemx_uint64_get_block, uint64_t) \
    FUNC(aclshmem_uint64_get_nbi, uint64_t) \
    FUNC(aclshmemx_uint64_get_nbi_warp, uint64_t) \
    FUNC(aclshmemx_uint64_get_nbi_block, uint64_t) \
    FUNC(aclshmem_char_get, unsigned char) \
    FUNC(aclshmemx_char_get_warp, unsigned char) \
    FUNC(aclshmemx_char_get_block, unsigned char) \
    FUNC(aclshmem_char_get_nbi, unsigned char) \
    FUNC(aclshmemx_char_get_nbi_warp, unsigned char) \
    FUNC(aclshmemx_char_get_nbi_block, unsigned char) \
    FUNC(aclshmem_bfloat16_get, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_get_warp, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_get_block, bfloat16_t) \
    FUNC(aclshmem_bfloat16_get_nbi, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_get_nbi_warp, bfloat16_t) \
    FUNC(aclshmemx_bfloat16_get_nbi_block, bfloat16_t) \
    FUNC(aclshmem_put8, void) \
    FUNC(aclshmemx_put8_warp, void) \
    FUNC(aclshmemx_put8_block, void) \
    FUNC(aclshmem_put8_nbi, void) \
    FUNC(aclshmemx_put8_nbi_warp, void) \
    FUNC(aclshmemx_put8_nbi_block, void) \
    FUNC(aclshmem_put16, void) \
    FUNC(aclshmemx_put16_warp, void) \
    FUNC(aclshmemx_put16_block, void) \
    FUNC(aclshmem_put16_nbi, void) \
    FUNC(aclshmemx_put16_nbi_warp, void) \
    FUNC(aclshmemx_put16_nbi_block, void) \
    FUNC(aclshmem_put32, void) \
    FUNC(aclshmemx_put32_warp, void) \
    FUNC(aclshmemx_put32_block, void) \
    FUNC(aclshmem_put32_nbi, void) \
    FUNC(aclshmemx_put32_nbi_warp, void) \
    FUNC(aclshmemx_put32_nbi_block, void) \
    FUNC(aclshmem_put64, void) \
    FUNC(aclshmemx_put64_warp, void) \
    FUNC(aclshmemx_put64_block, void) \
    FUNC(aclshmem_put64_nbi, void) \
    FUNC(aclshmemx_put64_nbi_warp, void) \
    FUNC(aclshmemx_put64_nbi_block, void) \
    FUNC(aclshmem_put128, void) \
    FUNC(aclshmemx_put128_warp, void) \
    FUNC(aclshmemx_put128_block, void) \
    FUNC(aclshmem_put128_nbi, void) \
    FUNC(aclshmemx_put128_nbi_warp, void) \
    FUNC(aclshmemx_put128_nbi_block, void) \
    FUNC(aclshmem_get8, void) \
    FUNC(aclshmemx_get8_warp, void) \
    FUNC(aclshmemx_get8_block, void) \
    FUNC(aclshmem_get8_nbi, void) \
    FUNC(aclshmemx_get8_nbi_warp, void) \
    FUNC(aclshmemx_get8_nbi_block, void) \
    FUNC(aclshmem_get16, void) \
    FUNC(aclshmemx_get16_warp, void) \
    FUNC(aclshmemx_get16_block, void) \
    FUNC(aclshmem_get16_nbi, void) \
    FUNC(aclshmemx_get16_nbi_warp, void) \
    FUNC(aclshmemx_get16_nbi_block, void) \
    FUNC(aclshmem_get32, void) \
    FUNC(aclshmemx_get32_warp, void) \
    FUNC(aclshmemx_get32_block, void) \
    FUNC(aclshmem_get32_nbi, void) \
    FUNC(aclshmemx_get32_nbi_warp, void) \
    FUNC(aclshmemx_get32_nbi_block, void) \
    FUNC(aclshmem_get64, void) \
    FUNC(aclshmemx_get64_warp, void) \
    FUNC(aclshmemx_get64_block, void) \
    FUNC(aclshmem_get64_nbi, void) \
    FUNC(aclshmemx_get64_nbi_warp, void) \
    FUNC(aclshmemx_get64_nbi_block, void) \
    FUNC(aclshmem_get128, void) \
    FUNC(aclshmemx_get128_warp, void) \
    FUNC(aclshmemx_get128_block, void) \
    FUNC(aclshmem_get128_nbi, void) \
    FUNC(aclshmemx_get128_nbi_warp, void) \
    FUNC(aclshmemx_get128_nbi_block, void) \
    FUNC(aclshmem_putmem, void) \
    FUNC(aclshmemx_putmem_warp, void) \
    FUNC(aclshmemx_putmem_block, void) \
    FUNC(aclshmem_putmem_nbi, void) \
    FUNC(aclshmemx_putmem_nbi_warp, void) \
    FUNC(aclshmemx_putmem_nbi_block, void) \
    FUNC(aclshmem_getmem, void) \
    FUNC(aclshmemx_getmem_warp, void) \
    FUNC(aclshmemx_getmem_block, void) \
    FUNC(aclshmem_getmem_nbi, void) \
    FUNC(aclshmemx_getmem_nbi_warp, void) \
    FUNC(aclshmemx_getmem_nbi_block, void)

#define SIMT_RMA_KERNEL(FUNCNAME, TYPE)                                                                                                         \
__simt_vf__ __launch_bounds__(1024) inline void simt_callee_call_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)   \
{                                                                                                                                               \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                                                  \
        printf("Testing " #FUNCNAME "\n");                                                                                                      \
    }                                                                                                                                           \
    asc_syncthreads();                                                                                                                          \
    simt::FUNCNAME((__gm__ TYPE*)dst, (__gm__ TYPE*)src, elem_size, pe);                                                                        \
}                                                                                                                                               \
__global__ __vector__ void kernel_call_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)                             \
{                                                                                                                                               \
    asc_vf_call<simt_callee_call_##FUNCNAME>(dim3(1024), dst, src, elem_size, pe);                                                              \
}                                                                                                                                               \
void test_##FUNCNAME(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)                                                    \
{                                                                                                                                               \
    kernel_call_##FUNCNAME<<<1, 0, stream>>>(dst, src, elem_size, pe);                                                                          \
}

SIMT_RMA_TEST(SIMT_RMA_KERNEL)
#undef SIMT_RMA_KERNEL
#undef SIMT_RMA_TEST

#define SIMT_RMA_SCHAR_TEST(FUNC) \
    FUNC(aclshmem_char_put, aclshmem_schar_put, signed char) \
    FUNC(aclshmemx_char_put_warp, aclshmemx_schar_put_warp, signed char) \
    FUNC(aclshmemx_char_put_block, aclshmemx_schar_put_block, signed char) \
    FUNC(aclshmem_char_put_nbi, aclshmem_schar_put_nbi, signed char) \
    FUNC(aclshmemx_char_put_nbi_warp, aclshmemx_schar_put_nbi_warp, signed char) \
    FUNC(aclshmemx_char_put_nbi_block, aclshmemx_schar_put_nbi_block, signed char) \
    FUNC(aclshmem_char_get, aclshmem_schar_get, signed char) \
    FUNC(aclshmemx_char_get_warp, aclshmemx_schar_get_warp, signed char) \
    FUNC(aclshmemx_char_get_block, aclshmemx_schar_get_block, signed char) \
    FUNC(aclshmem_char_get_nbi, aclshmem_schar_get_nbi, signed char) \
    FUNC(aclshmemx_char_get_nbi_warp, aclshmemx_schar_get_nbi_warp, signed char) \
    FUNC(aclshmemx_char_get_nbi_block, aclshmemx_schar_get_nbi_block, signed char)

#define SIMT_RMA_SCHAR_KERNEL(FUNCNAME, REPRNAME, TYPE)                                                                                         \
__simt_vf__ __launch_bounds__(1024) inline void simt_callee_call_##REPRNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)   \
{                                                                                                                                               \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                                                  \
        printf("Testing " #REPRNAME "\n");                                                                                                      \
    }                                                                                                                                           \
    asc_syncthreads();                                                                                                                          \
    simt::FUNCNAME((__gm__ TYPE*)dst, (__gm__ TYPE*)src, elem_size, pe);                                                                        \
}                                                                                                                                               \
__global__ __vector__ void kernel_call_##REPRNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)                             \
{                                                                                                                                               \
    asc_vf_call<simt_callee_call_##REPRNAME>(dim3(1024), dst, src, elem_size, pe);                                                              \
}                                                                                                                                               \
void test_##REPRNAME(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)                                                    \
{                                                                                                                                               \
    kernel_call_##REPRNAME<<<1, 0, stream>>>(dst, src, elem_size, pe);                                                                          \
}

SIMT_RMA_SCHAR_TEST(SIMT_RMA_SCHAR_KERNEL)
#undef SIMT_RMA_SCHAR_KERNEL
#undef SIMT_RMA_SCHAR_TEST

// ===================_p functions==================================

#define SIMT_RMA_SCALAR_PUT_TEST(FUNC) \
    FUNC(aclshmem_half_p, half) \
    FUNC(aclshmem_float_p, float) \
    FUNC(aclshmem_int8_p, int8_t) \
    FUNC(aclshmem_int16_p, int16_t) \
    FUNC(aclshmem_int32_p, int32_t) \
    FUNC(aclshmem_int64_p, int64_t) \
    FUNC(aclshmem_uint8_p, uint8_t) \
    FUNC(aclshmem_uint16_p, uint16_t) \
    FUNC(aclshmem_uint32_p, uint32_t) \
    FUNC(aclshmem_uint64_p, uint64_t) \
    FUNC(aclshmem_char_p, unsigned char) \
    FUNC(aclshmem_bfloat16_p, bfloat16_t)

#define SIMT_RMA_SCALAR_PUT_KERNEL(FUNCNAME, TYPE)                                                                                              \
__simt_vf__ __launch_bounds__(1024) inline void simt_callee_call_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)   \
{                                                                                                                                               \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                                                  \
        printf("Testing " #FUNCNAME "\n");                                                                                                      \
    }                                                                                                                                           \
    asc_syncthreads();                                                                                                                          \
    __gm__ TYPE* dst_ptr = (__gm__ TYPE*)dst;                                                                                                   \
    __gm__ TYPE* src_ptr = (__gm__ TYPE*)src;                                                                                                   \
    TYPE val = *src_ptr;                                                                                                                        \
    simt::FUNCNAME((__gm__ TYPE*)dst, val, pe);                                                                                                 \
}                                                                                                                                               \
__global__ __vector__ void kernel_call_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)                             \
{                                                                                                                                               \
    asc_vf_call<simt_callee_call_##FUNCNAME>(dim3(1), dst, src, elem_size, pe);                                                                 \
}                                                                                                                                               \
void test_##FUNCNAME(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)                                                    \
{                                                                                                                                               \
    kernel_call_##FUNCNAME<<<1, 0, stream>>>(dst, src, elem_size, pe);                                                                          \
}

SIMT_RMA_SCALAR_PUT_TEST(SIMT_RMA_SCALAR_PUT_KERNEL)
#undef SIMT_RMA_SCALAR_PUT_KERNEL

__simt_vf__ __launch_bounds__(1024) inline void simt_callee_call_aclshmem_schar_p(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Testing " "aclshmem_schar_p" "\n");
    }
    asc_syncthreads();
    __gm__ signed char* dst_ptr = (__gm__ signed char*)dst;
    __gm__ signed char* src_ptr = (__gm__ signed char*)src;
    signed char val = *src_ptr;
    simt::aclshmem_char_p((__gm__ signed char*)dst, val, pe);
}
__global__ __vector__ void kernel_call_aclshmem_schar_p(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)
{
    asc_vf_call<simt_callee_call_aclshmem_schar_p>(dim3(1), dst, src, elem_size, pe);
}
void test_aclshmem_schar_p(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)
{
    kernel_call_aclshmem_schar_p<<<1, 0, stream>>>(dst, src, elem_size, pe);
}

// ===================_g functions==================================

#define SIMT_RMA_SCALAR_GET_TEST(FUNC) \
    FUNC(aclshmem_half_g, half) \
    FUNC(aclshmem_float_g, float) \
    FUNC(aclshmem_int8_g, int8_t) \
    FUNC(aclshmem_int16_g, int16_t) \
    FUNC(aclshmem_int32_g, int32_t) \
    FUNC(aclshmem_int64_g, int64_t) \
    FUNC(aclshmem_uint8_g, uint8_t) \
    FUNC(aclshmem_uint16_g, uint16_t) \
    FUNC(aclshmem_uint32_g, uint32_t) \
    FUNC(aclshmem_uint64_g, uint64_t) \
    FUNC(aclshmem_char_g, unsigned char) \
    FUNC(aclshmem_bfloat16_g, bfloat16_t)

#define SIMT_RMA_SCALAR_GET_KERNEL(FUNCNAME, TYPE)                                                                                              \
__simt_vf__ __launch_bounds__(1024) inline void simt_callee_call_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)   \
{                                                                                                                                               \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                                                                                                  \
        printf("Testing " #FUNCNAME "\n");                                                                                                      \
    }                                                                                                                                           \
    asc_syncthreads();                                                                                                                          \
    TYPE val = simt::FUNCNAME((__gm__ TYPE*)src, pe);                                                                                           \
    __gm__ TYPE* dst_ptr = (__gm__ TYPE*)dst;                                                                                                   \
    *dst_ptr = val;                                                                                                                             \
}                                                                                                                                               \
__global__ __vector__ void kernel_call_##FUNCNAME(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)                             \
{                                                                                                                                               \
    asc_vf_call<simt_callee_call_##FUNCNAME>(dim3(1), dst, src, elem_size, pe);                                                                 \
}                                                                                                                                               \
void test_##FUNCNAME(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)                                                    \
{                                                                                                                                               \
    kernel_call_##FUNCNAME<<<1, 0, stream>>>(dst, src, elem_size, pe);                                                                          \
}
SIMT_RMA_SCALAR_GET_TEST(SIMT_RMA_SCALAR_GET_KERNEL)
#undef SIMT_RMA_SCALAR_GET_KERNEL

__simt_vf__ __launch_bounds__(1024) inline void simt_callee_call_aclshmem_schar_g(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Testing " "aclshmem_schar_g" "\n");
    }
    asc_syncthreads();
    signed char val = simt::aclshmem_char_g((__gm__ signed char*)src, pe);
    __gm__ signed char* dst_ptr = (__gm__ signed char*)dst;
    *dst_ptr = val;
}
__global__ __vector__ void kernel_call_aclshmem_schar_g(__gm__ void* dst, __gm__ void* src, size_t elem_size, int32_t pe)
{
    asc_vf_call<simt_callee_call_aclshmem_schar_g>(dim3(1), dst, src, elem_size, pe);
}
void test_aclshmem_schar_g(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe)
{
    kernel_call_aclshmem_schar_g<<<1, 0, stream>>>(dst, src, elem_size, pe);
}
#endif // defined(ACLSHMEM_SIMT_SUPPORT)