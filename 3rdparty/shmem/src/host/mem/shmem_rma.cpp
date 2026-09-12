/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "host/data_plane/shmem_host_so.h"
#include "gm2gm/shmemi_device_rma.h"
#include "host_device/shmem_common_types.h"
#include "gm2gm/shmemi_device_cc_kernel.h"

using namespace std;

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                                    \
    /**                                                                                                                 \
     * @brief Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.       \
     *                                                                                                                  \
     * @param dst               [in] Pointer on local device of the destination data.                                   \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                    \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                                  \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                                \
     * @param signal            [in] The value used to update sig_addr.                                                 \
     * @param sig_op            [in] Operation used to update sig_addr with signal.                                     \
     *                          Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                           \
     * @param pe                [in] PE number of the remote PE.                                                        \
     */                                                                                                                 \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_put_signal(TYPE *dst, TYPE *src, size_t elem_size, uint8_t *sig_addr,      \
                                                      int32_t signal, int sig_op, int pe)                               \
    {                                                                                                                   \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put_" #NAME "_mem_signal", ACLSHMEMI_OP_PUT_SIGNAL, NO_NBI,  \
                                              (uint8_t *)dst, (uint8_t *)src, elem_size, sizeof(TYPE), pe, sig_addr,    \
                                              signal, sig_op, 1, 1, g_state_host.default_stream,                        \
                                              g_state_host.default_block_num);                                          \
        if (ret < 0) {                                                                                                  \
            SHM_LOG_ERROR("device calling transfer failed");                                                            \
        }                                                                                                               \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL)
#undef ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI(NAME, TYPE)                                                                    \
    /**                                                                                                                     \
     * @brief Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.          \
     *                                                                                                                      \
     * @param dst               [in] Pointer on local device of the destination data.                                       \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                        \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                                      \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                                    \
     * @param signal            [in] The value used to update sig_addr.                                                     \
     * @param sig_op            [in] Operation used to update sig_addr with signal.                                         \
     *                          Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                               \
     * @param pe                [in] PE number of the remote PE.                                                            \
     */                                                                                                                     \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_put_signal_nbi(TYPE *dst, TYPE *src, size_t elem_size, uint8_t *sig_addr,      \
                                                          int32_t signal, int sig_op, int pe)                               \
    {                                                                                                                       \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put_" #NAME "_mem_signal_nbi", ACLSHMEMI_OP_PUT_SIGNAL, NBI,     \
                                              (uint8_t *)dst, (uint8_t *)src, elem_size, sizeof(TYPE), pe, sig_addr,        \
                                              signal, sig_op, 1, 1, g_state_host.default_stream,                            \
                                              g_state_host.default_block_num);                                              \
        if (ret < 0) {                                                                                                      \
            SHM_LOG_ERROR("device calling transfer failed");                                                                \
        }                                                                                                                   \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI)
#undef ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI

#define ACLSHMEM_PUT_SIZE_MEM_SIGNAL(BITS)                                                                             \
    /**                                                                                                                \
     * @brief Synchronous interface. Copy a contiguous data from local to symmetric address on the specified PE and    \
     *        updating a remote signal flag on completion.                                                             \
     *                                                                                                                 \
     * @param dst               [in] Pointer on local device of the destination data.                                  \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                   \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                                 \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                               \
     * @param signal            [in] The value used to update sig_addr.                                                \
     * @param sig_op            [in] Operation used to update sig_addr with signal.                                    \
     *                               Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                     \
     * @param pe                [in] PE number of the remote PE.                                                       \
     */                                                                                                                \
    ACLSHMEM_HOST_API void aclshmem_put##BITS##_signal(void *dst, void *src, size_t nelems,                            \
                                                       int32_t *sig_addr, int32_t signal, int sig_op, int pe)          \
    {                                                                                                                  \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put" #BITS "_signal", ACLSHMEMI_OP_PUT_SIGNAL, NO_NBI,      \
                                                 (uint8_t *)dst, (uint8_t *)src, nelems * (BITS / 8), 1, pe,           \
                                                 (uint8_t *)sig_addr, signal, sig_op, 1, 1,                            \
                                                 g_state_host.default_stream, g_state_host.default_block_num);         \
        if (ret < 0) {                                                                                                 \
            SHM_LOG_ERROR("device calling transfer failed.");                                                          \
        }                                                                                                              \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM_SIGNAL)
#undef ACLSHMEM_PUT_SIZE_MEM_SIGNAL

#define ACLSHMEM_PUT_SIZE_MEM_SIGNAL_NBI(BITS)                                                                        \
    /**                                                                                                               \
     * @brief Asynchronous interface. Copy a contiguous data from local to symmetric address on the specified PE and  \
     *        updating a remote signal flag on completion.                                                            \
     *                                                                                                                \
     * @param dst               [in] Pointer on local device of the destination data.                                 \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                  \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                                \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                              \
     * @param signal            [in] The value used to update sig_addr.                                               \
     * @param sig_op            [in] Operation used to update sig_addr with signal.                                   \
     *                               Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                    \
     * @param pe                [in] PE number of the remote PE.                                                      \
     */                                                                                                               \
    ACLSHMEM_HOST_API void aclshmem_put##BITS##_signal_nbi(void *dst, void *src, size_t nelems,                       \
                                                           int32_t *sig_addr, int32_t signal, int sig_op, int pe)     \
    {                                                                                                                 \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put" #BITS "_signal_nbi", ACLSHMEMI_OP_PUT_SIGNAL, NBI,    \
                                                 (uint8_t *)dst, (uint8_t *)src, nelems * (BITS / 8), 1, pe,          \
                                                 (uint8_t *)sig_addr, signal, sig_op, 1, 1,                           \
                                                 g_state_host.default_stream, g_state_host.default_block_num);        \
        if (ret < 0) {                                                                                                \
            SHM_LOG_ERROR("device calling transfer failed.");                                                         \
        }                                                                                                             \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_MEM_SIGNAL_NBI)
#undef ACLSHMEM_PUT_SIZE_MEM_SIGNAL_NBI

void aclshmemx_putmem_signal_nbi(void *dst, void *src, size_t elem_size, void *sig_addr, int32_t signal, int sig_op, int pe)
{
    int ret = aclshmemi_prepare_and_post_rma("aclshmemx_putmem_signal_nbi", ACLSHMEMI_OP_PUT_SIGNAL, NBI, (uint8_t *)dst,
                                          (uint8_t *)src, elem_size, 1, pe, (uint8_t *)sig_addr, signal, sig_op, 1, 1,
                                          g_state_host.default_stream, g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("device calling transfer failed");
    }
}

void aclshmemx_putmem_signal(void *dst, void *src, size_t elem_size, void *sig_addr, int32_t signal, int sig_op, int pe)
{
    int ret = aclshmemi_prepare_and_post_rma("aclshmemx_putmem_signal", ACLSHMEMI_OP_PUT_SIGNAL, NO_NBI, (uint8_t *)dst,
                                          (uint8_t *)src, elem_size, 1, pe, (uint8_t *)sig_addr, signal, sig_op, 1, 1,
                                          g_state_host.default_stream, g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("device calling transfer failed");
    }
}

void aclshmemx_signal_op_on_stream(int32_t *sig_addr, int32_t signal, int sig_op, int pe, aclrtStream stream)
{
    if (stream == nullptr) {
        stream = g_state_host.default_stream;
    }
    call_aclshmemi_signal_op_on_stream_kernel(sig_addr, signal, sig_op, pe, stream);
}