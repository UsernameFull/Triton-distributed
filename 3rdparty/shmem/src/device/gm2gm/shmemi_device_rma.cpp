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
#include "shmemi_device_rma.h"
#include "host_device/shmem_common_types.h"
#include "shmemi_device_common.h"
#include "gm2gm/engine/shmem_device_rdma.hpp"
#include "device/gm2gm/shmem_device_rma.h"
#include "shmemi_device_mo.h"

using namespace std;

#define ACLSHMEM_TYPE_FUNC(FUNC) \
    FUNC(float, float);       \
    FUNC(double, double);     \
    FUNC(int8, int8_t);       \
    FUNC(int16, int16_t);     \
    FUNC(int32, int32_t);     \
    FUNC(int64, int64_t);     \
    FUNC(uint8, uint8_t);     \
    FUNC(uint16, uint16_t);   \
    FUNC(uint32, uint32_t);   \
    FUNC(uint64, uint64_t);   \
    FUNC(char, char)

// kernels
ACLSHMEM_GLOBAL void aclshmemi_putmem_nbi(GM_ADDR dst, GM_ADDR src, uint32_t elem_size, int32_t pe)
{
    aclshmem_uint8_put_nbi(dst, src, elem_size, pe);
}

ACLSHMEM_GLOBAL void aclshmemi_putmem(GM_ADDR dst, GM_ADDR src, uint32_t elem_size, int32_t pe)
{
    aclshmem_uint8_put(dst, src, elem_size, pe);
}

ACLSHMEM_GLOBAL void aclshmemi_putbits(GM_ADDR dst, GM_ADDR src, non_contiguous_copy_param copy_params, int32_t pe)
{
    switch (copy_params.length) {
        case 1:
            aclshmem_iput8(dst, src, copy_params.dst_ld, copy_params.src_ld, copy_params.repeat, pe);
            break;
        case 2:
            aclshmem_iput16(dst, src, copy_params.dst_ld, copy_params.src_ld, copy_params.repeat, pe);
            break;
        case 4:
            aclshmem_iput32(dst, src, copy_params.dst_ld, copy_params.src_ld, copy_params.repeat, pe);
            break;
        case 8:
            aclshmem_iput64(dst, src, copy_params.dst_ld, copy_params.src_ld, copy_params.repeat, pe);
            break;
        case 16:
            aclshmem_iput128(dst, src, copy_params.dst_ld, copy_params.src_ld, copy_params.repeat, pe);
            break;
        default:
            break;
    }
}

ACLSHMEM_GLOBAL void aclshmemi_getmem_nbi(GM_ADDR dst, GM_ADDR src, uint32_t elem_size, int32_t pe)
{
    aclshmem_uint8_get_nbi(dst, src, elem_size, pe);
}

ACLSHMEM_GLOBAL void aclshmemi_getmem(GM_ADDR dst, GM_ADDR src, uint32_t elem_size, int32_t pe)
{
    aclshmem_uint8_get(dst, src, elem_size, pe);
}

ACLSHMEM_GLOBAL void aclshmemi_getbits(GM_ADDR dst, GM_ADDR src, non_contiguous_copy_param copy_params, int32_t pe)
{
    switch (copy_params.length) {
        case 1:
            aclshmem_iget8(dst, src, copy_params.dst_ld, copy_params.src_ld, copy_params.repeat, pe);
            break;
        case 2:
            aclshmem_iget16(dst, src, copy_params.dst_ld, copy_params.src_ld, copy_params.repeat, pe);
            break;
        case 4:
            aclshmem_iget32(dst, src, copy_params.dst_ld, copy_params.src_ld, copy_params.repeat, pe);
            break;
        case 8:
            aclshmem_iget64(dst, src, copy_params.dst_ld, copy_params.src_ld, copy_params.repeat, pe);
            break;
        case 16:
            aclshmem_iget128(dst, src, copy_params.dst_ld, copy_params.src_ld, copy_params.repeat, pe);
            break;
        default:
            break;
    }
}

ACLSHMEM_GLOBAL_VECTOR void aclshmemi_putmem_signal(GM_ADDR dst, GM_ADDR src, uint32_t elem_size, GM_ADDR sig_addr, int32_t signal,
                                       int sig_op, int pe)
{
    __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);
    aclshmem_uint8_put_signal(dst, src, elem_size, sig_addr_int32, signal, sig_op, pe);
}

ACLSHMEM_GLOBAL_VECTOR void aclshmemi_putmem_signal_nbi(GM_ADDR dst, GM_ADDR src, uint32_t elem_size, GM_ADDR sig_addr,
                                           int32_t signal, int sig_op, int pe)
{
    __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);
    aclshmem_uint8_put_signal_nbi(dst, src, elem_size, sig_addr_int32, signal, sig_op, pe);
}

ACLSHMEM_GLOBAL void k_aclshmem_getmem(GM_ADDR dst, GM_ADDR src, size_t elem_size, int32_t pe)
{
    aclshmem_getmem(dst, src, elem_size, pe);
}

ACLSHMEM_GLOBAL void k_aclshmemi_quiet()
{
    aclshmemi_quiet();
}

// kernel function calling entrance
int32_t aclshmemi_prepare_and_post_rma(const char *api_name, aclshmemi_op_t desc, bool is_nbi, uint8_t *dst, uint8_t *src,
                                    size_t n_elems, size_t elem_bytes, int pe, uint8_t *sig_addr, int32_t signal,
                                    int sig_op, ptrdiff_t lstride, ptrdiff_t rstride, aclrtStream acl_strm,
                                    size_t block_size)
{
    if (is_nbi) {
        switch (desc) {
            case ACLSHMEMI_OP_PUT:
                aclshmemi_putmem_nbi<<<block_size, 0, acl_strm>>>(dst, src, n_elems * elem_bytes, pe);
                break;
            case ACLSHMEMI_OP_GET:
                aclshmemi_getmem_nbi<<<block_size, 0, acl_strm>>>(dst, src, n_elems * elem_bytes, pe);
                break;
            case ACLSHMEMI_OP_PUT_SIGNAL:
                aclshmemi_putmem_signal_nbi<<<block_size, 0, acl_strm>>>(dst, src, n_elems * elem_bytes, sig_addr,
                                                                      signal, sig_op, pe);
                break;
            case ACLSHMEMI_OP_QUIET:
                k_aclshmemi_quiet<<<block_size, 0, acl_strm>>>();
                break;
            default:
                break;
        }
    } else {
        bool contiguous = lstride == 1 && rstride == 1;
        if (contiguous) {
            switch (desc) {
                case ACLSHMEMI_OP_PUT:
                    aclshmemi_putmem<<<block_size, 0, acl_strm>>>(dst, src, n_elems * elem_bytes, pe);
                    break;
                case ACLSHMEMI_OP_GET:
                    aclshmemi_getmem<<<block_size, 0, acl_strm>>>(dst, src, n_elems * elem_bytes, pe);
                    break;
                case ACLSHMEMI_OP_PUT_SIGNAL:
                    aclshmemi_putmem_signal<<<block_size, 0, acl_strm>>>(dst, src, n_elems * elem_bytes, sig_addr, signal,
                                                                    sig_op, pe);
                    break;
                case ACLSHMEMI_OP_QUIET:
                    k_aclshmemi_quiet<<<block_size, 0, acl_strm>>>();
                    break;
                default:
                    break;
            }
        } else {
            non_contiguous_copy_param copy_params {static_cast<uint32_t>(n_elems),
                                                   static_cast<uint32_t>(elem_bytes),
                                                   static_cast<uint32_t>(rstride),
                                                   static_cast<uint32_t>(lstride)};
            switch (desc) {
                case ACLSHMEMI_OP_PUT:
                    aclshmemi_putbits<<<block_size, 0, acl_strm>>>(dst, src, copy_params, pe);
                    break;
                case ACLSHMEMI_OP_GET:
                    aclshmemi_getbits<<<block_size, 0, acl_strm>>>(dst, src, copy_params, pe);
                    break;
                case ACLSHMEMI_OP_QUIET:
                    k_aclshmemi_quiet<<<block_size, 0, acl_strm>>>();
                    break;
                default:
                    break;
            }
        }
    }
    return 0;
}

#define ACLSHMEMI_TYPENAME_P(NAME, TYPE)                                                \
    ACLSHMEM_GLOBAL void aclshmemi_##NAME##_p(GM_ADDR dest_addr, const TYPE value, int pe) \
    {                                                                                \
        __gm__ TYPE *dst = (__gm__ TYPE *)dest_addr;                                 \
        aclshmem_##NAME##_p(dst, value, pe);                                            \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEMI_TYPENAME_P)

#undef ACLSHMEMI_TYPENAME_P

// aclshmem_g
#define ACLSHMEMI_TYPENAME_G(NAME, TYPE)                                            \
    ACLSHMEM_GLOBAL void aclshmemi_##NAME##_g(GM_ADDR src, int pe, GM_ADDR value_addr) \
    {                                                                            \
        __gm__ TYPE *src_addr = (__gm__ TYPE *)src;                              \
        __gm__ TYPE *dst_addr = (__gm__ TYPE *)value_addr;                       \
        *dst_addr = aclshmem_##NAME##_g(src_addr, pe);                              \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEMI_TYPENAME_G)
#undef ACLSHMEMI_TYPENAME_G

#define ACLSHMEMI_TYPENAME_PREPARE_RMA_P(NAME, TYPE)                                                           \
    void aclshmemi_prepare_and_post_rma_##NAME##_p(const char *api_name, uint8_t *dst_ptr, TYPE value, int pe, \
                                                aclrtStream acl_strm, size_t block_size)                    \
    {                                                                                                       \
        aclshmemi_##NAME##_p<<<block_size, 0, acl_strm>>>(dst_ptr, value, pe);                                 \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEMI_TYPENAME_PREPARE_RMA_P)
#undef ACLSHMEMI_TYPENAME_PREPARE_RMA_P
