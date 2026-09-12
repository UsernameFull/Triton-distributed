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
#include "gm2gm/shmemi_device_rma.h"
#include "host_device/shmem_common_types.h"
#include "host/data_plane/shmem_host_rma.h"

using namespace std;

inline bool check_heap_addr(void *ptr, uint64_t heap_base, size_t heap_size)
{
    uint64_t lower_bound = heap_base;
    uint64_t upper_bound = lower_bound + heap_size;
    if (uint64_t(ptr) < lower_bound || uint64_t(ptr) >= upper_bound) {
        return false;
    }
    return true;
}
inline bool is_host_mem_heap(void *ptr)
{
    if (g_state.host_heap_base == nullptr) {
        return false;
    }
    return check_heap_addr(ptr, (uint64_t)g_state.host_heap_base, g_state.heap_size);
}

void *aclshmem_ptr(void *ptr, int32_t pe)
{
    if (pe < 0 || pe >= aclshmem_n_pes()) {
        SHM_LOG_ERROR("aclshmem_ptr Failed. PE: " << aclshmem_my_pe() << " Got Illegal PE !!");
        return nullptr;
    }
    uint64_t heap_base = is_host_mem_heap(ptr) ? (uint64_t)g_state.host_heap_base : (uint64_t)g_state.heap_base;
    if (!check_heap_addr(ptr, heap_base, g_state.heap_size)) {
        SHM_LOG_ERROR("aclshmem_ptr Failed. PE: " << aclshmem_my_pe() << " Got Illegal Address !!");
        return nullptr;
    }

    uint64_t offset = (uint64_t)ptr - heap_base;
    void *symm_ptr = is_host_mem_heap(ptr) ? g_state.p2p_host_heap_base[pe] : g_state.p2p_device_heap_base[pe];
    if (symm_ptr != nullptr) {
        symm_ptr = reinterpret_cast<void*>(reinterpret_cast<uint64_t>(symm_ptr) + offset);
        return symm_ptr;
    }
    SHM_LOG_ERROR("aclshmem_ptr Failed. PE: " << aclshmem_my_pe()
        << " g_state.p2p_" << (is_host_mem_heap(ptr) ? "host" : "device") << "_heap_base contains nullptr, Please Check Init Status!!");
    return nullptr;
}

// Set Memcpy Interfaces necessary UB Buffer.
int32_t aclshmemx_set_mte_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id)
{
    g_state.mte_config.aclshmem_ub = offset;
    g_state.mte_config.ub_size = ub_size;
    g_state.mte_config.sync_id = sync_id;
    ACLSHMEM_CHECK_RET(update_device_state());
    return ACLSHMEM_SUCCESS;
}

// Set SDMA Interfaces necessary UB Buffer.
int32_t aclshmemx_set_sdma_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id)
{
    g_state.sdma_config.aclshmem_ub = offset;
    g_state.sdma_config.ub_size = ub_size;
    g_state.sdma_config.sync_id = sync_id;
    ACLSHMEM_CHECK_RET(update_device_state());
    return ACLSHMEM_SUCCESS;
}

// Set RDMA Interfaces necessary UB Buffer.
int32_t aclshmemx_set_rdma_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id)
{
    g_state.rdma_config.aclshmem_ub = offset;
    g_state.rdma_config.ub_size = ub_size;
    g_state.rdma_config.sync_id = sync_id;
    ACLSHMEM_CHECK_RET(update_device_state());
    return ACLSHMEM_SUCCESS;
}

#define ACLSHMEM_TYPE_PUT(NAME, TYPE)                                                                                    \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_put(TYPE *dest, TYPE *source, size_t nelems, int pe)                        \
    {                                                                                                                    \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put_" #NAME "_mem", ACLSHMEMI_OP_PUT, NO_NBI, (uint8_t *)dest,\
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,          \
                                              g_state_host.default_stream, g_state_host.default_block_num);              \
        if (ret < 0) {                                                                                                   \
            SHM_LOG_ERROR("device calling transfer failed");                                                             \
        }                                                                                                                \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_PUT)
#undef ACLSHMEM_TYPE_PUT

#define ACLSHMEM_TYPE_PUT_NBI(NAME, TYPE)                                                                                   \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_put_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe)                       \
    {                                                                                                                       \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put_" #NAME "_mem_nbi", ACLSHMEMI_OP_PUT, NBI, (uint8_t *)dest,  \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,             \
                                              g_state_host.default_stream, g_state_host.default_block_num);                 \
        if (ret < 0) {                                                                                                      \
            SHM_LOG_ERROR("device calling transfer failed");                                                                \
        }                                                                                                                   \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_PUT_NBI)
#undef ACLSHMEM_TYPE_PUT_NBI

#define ACLSHMEM_TYPE_IPUT(NAME, TYPE)                                                                                           \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_iput(TYPE *dest, TYPE *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) \
    {                                                                                                                            \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_iput_" #NAME "", ACLSHMEMI_OP_PUT, NO_NBI, (uint8_t *)dest,           \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, dst, sst,              \
                                              g_state_host.default_stream, g_state_host.default_block_num);                      \
        if (ret < 0) {                                                                                                           \
            SHM_LOG_ERROR("device calling transfer failed");                                                                     \
        }                                                                                                                        \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_IPUT);
#undef ACLSHMEM_TYPE_IPUT

#define ACLSHMEM_PUT_SIZE(BITS)                                                                                        \
    ACLSHMEM_HOST_API void aclshmem_put##BITS(void *dst, void *src, uint32_t elem_size, int32_t pe)                    \
    {                                                                                                                  \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put_" #BITS "", ACLSHMEMI_OP_PUT, NO_NBI, (uint8_t *)dst,   \
                                              (uint8_t *)src, elem_size, (BITS) / 8, pe, nullptr, 0, 0, 1, 1,          \
                                              g_state_host.default_stream, g_state_host.default_block_num);            \
        if (ret < 0) {                                                                                                 \
            SHM_LOG_ERROR("device calling transfer failed");                                                           \
        }                                                                                                              \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE);
#undef ACLSHMEM_PUT_SIZE

#define ACLSHMEM_PUT_SIZE_NBI(BITS)                                                                                    \
    ACLSHMEM_HOST_API void aclshmem_put##BITS##_nbi(void *dst, void *src, uint32_t elem_size, int32_t pe)              \
    {                                                                                                                  \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put_" #BITS "_nbi", ACLSHMEMI_OP_PUT, NBI, (uint8_t *)dst,  \
                                              (uint8_t *)src, elem_size, (BITS) / 8, pe, nullptr, 0, 0, 1, 1,          \
                                              g_state_host.default_stream, g_state_host.default_block_num);            \
        if (ret < 0) {                                                                                                 \
            SHM_LOG_ERROR("device calling transfer failed");                                                           \
        }                                                                                                              \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_PUT_SIZE_NBI);
#undef ACLSHMEM_PUT_SIZE_NBI

#define ACLSHMEM_IPUT_SIZE(BITS)                                                                                              \
    ACLSHMEM_HOST_API void aclshmem_iput##BITS(void *dest, void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) \
    {                                                                                                                         \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_iput_" #BITS "", ACLSHMEMI_OP_PUT, NO_NBI, (uint8_t *)dest,        \
                                              (uint8_t *)source, nelems, (BITS) / 8, pe, nullptr, 0, 0, dst, sst,             \
                                              g_state_host.default_stream, g_state_host.default_block_num);                   \
        if (ret < 0) {                                                                                                        \
            SHM_LOG_ERROR("device calling transfer failed");                                                                  \
        }                                                                                                                     \
    }
ACLSHMEM_SIZE_FUNC(ACLSHMEM_IPUT_SIZE);
#undef ACLSHMEM_IPUT_SIZE

#define ACLSHMEM_TYPE_GET(NAME, TYPE)                                                                                    \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_get(TYPE *dest, TYPE *source, size_t nelems, int pe)                        \
    {                                                                                                                    \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_get_" #NAME "_mem", ACLSHMEMI_OP_GET, NO_NBI, (uint8_t *)dest,\
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,          \
                                              g_state_host.default_stream, g_state_host.default_block_num);              \
        if (ret < 0) {                                                                                                   \
            SHM_LOG_ERROR("device calling transfer failed");                                                             \
        }                                                                                                                \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_GET)
#undef ACLSHMEM_TYPE_GET

#define ACLSHMEM_TYPE_GET_NBI(NAME, TYPE)                                                                                   \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_get_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe)                       \
    {                                                                                                                       \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_get_" #NAME "_mem_nbi", ACLSHMEMI_OP_GET, NBI, (uint8_t *)dest,  \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,             \
                                              g_state_host.default_stream, g_state_host.default_block_num);                 \
        if (ret < 0) {                                                                                                      \
            SHM_LOG_ERROR("device calling transfer failed");                                                                \
        }                                                                                                                   \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_GET_NBI)
#undef ACLSHMEM_TYPE_GET_NBI

#define ACLSHMEM_TYPE_IGET(NAME, TYPE)                                                                                           \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_iget(TYPE *dest, TYPE *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) \
    {                                                                                                                            \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_iget_" #NAME "", ACLSHMEMI_OP_GET, NO_NBI, (uint8_t *)dest,           \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, dst, sst,              \
                                              g_state_host.default_stream, g_state_host.default_block_num);                      \
        if (ret < 0) {                                                                                                           \
            SHM_LOG_ERROR("device calling transfer failed");                                                                     \
        }                                                                                                                        \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_IGET);
#undef ACLSHMEM_TYPE_IGET

#define ACLSHMEM_GET_SIZE(BITS)                                                                                        \
    ACLSHMEM_HOST_API void aclshmem_get##BITS(void *dst, void *src, uint32_t elem_size, int32_t pe)                    \
    {                                                                                                                  \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_get_" #BITS "", ACLSHMEMI_OP_GET, NO_NBI, (uint8_t *)dst,   \
                                              (uint8_t *)src, elem_size, (BITS) / 8, pe, nullptr, 0, 0, 1, 1,          \
                                              g_state_host.default_stream, g_state_host.default_block_num);            \
        if (ret < 0) {                                                                                                 \
            SHM_LOG_ERROR("device calling transfer failed");                                                           \
        }                                                                                                              \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_GET_SIZE);
#undef ACLSHMEM_GET_SIZE

#define ACLSHMEM_GET_SIZE_NBI(BITS)                                                                                    \
    ACLSHMEM_HOST_API void aclshmem_get##BITS##_nbi(void *dst, void *src, uint32_t elem_size, int32_t pe)              \
    {                                                                                                                  \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_get_" #BITS "_nbi", ACLSHMEMI_OP_GET, NBI, (uint8_t *)dst,  \
                                              (uint8_t *)src, elem_size, (BITS) / 8, pe, nullptr, 0, 0, 1, 1,          \
                                              g_state_host.default_stream, g_state_host.default_block_num);            \
        if (ret < 0) {                                                                                                 \
            SHM_LOG_ERROR("device calling transfer failed");                                                           \
        }                                                                                                              \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_GET_SIZE_NBI);
#undef ACLSHMEM_GET_SIZE_NBI

#define ACLSHMEM_IGET_SIZE(BITS)                                                                                              \
    ACLSHMEM_HOST_API void aclshmem_iget##BITS(void *dest, void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) \
    {                                                                                                                         \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_iget_" #BITS "", ACLSHMEMI_OP_GET, NO_NBI, (uint8_t *)dest,        \
                                              (uint8_t *)source, nelems, (BITS) / 8, pe, nullptr, 0, 0, dst, sst,             \
                                              g_state_host.default_stream, g_state_host.default_block_num);                   \
        if (ret < 0) {                                                                                                        \
            SHM_LOG_ERROR("device calling transfer failed");                                                                  \
        }                                                                                                                     \
    }

ACLSHMEM_SIZE_FUNC(ACLSHMEM_IGET_SIZE);
#undef ACLSHMEM_IGET_SIZE

#define ACLSHMEM_TYPENAME_P(NAME, TYPE)                                                                                \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_p(TYPE *dst, const TYPE value, int pe)                                    \
    {                                                                                                                  \
        aclshmemi_prepare_and_post_rma_##NAME##_p("aclshmem_" #NAME "_p", (uint8_t *)dst, value, pe,                   \
                                               g_state_host.default_stream, g_state_host.default_block_num);           \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_P)
#undef ACLSHMEM_TYPENAME_P

#define ACLSHMEM_TYPENAME_G(NAME, TYPE)                                                                                \
    ACLSHMEM_HOST_API TYPE aclshmem_##NAME##_g(TYPE *src, int32_t pe)                                                  \
    {                                                                                                                  \
        TYPE value {};                                                                                                 \
        auto ptr = aclshmem_ptr(src, pe);                                                                              \
        if (ptr == nullptr) {                                                                                          \
            SHM_LOG_ERROR("aclshmem_g failed");                                                                        \
            return value;                                                                                              \
        }                                                                                                              \
        int ret =                                                                                                      \
            aclrtMemcpy(&value, sizeof(TYPE), reinterpret_cast<void *>(ptr), sizeof(TYPE), ACL_MEMCPY_DEVICE_TO_HOST); \
        if (ret != 0) {                                                                                                \
            SHM_LOG_ERROR("aclshmem_g failed");                                                                        \
        }                                                                                                              \
        return value;                                                                                                  \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_G)
#undef ACLSHMEM_TYPENAME_G

void aclshmem_putmem(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = aclshmemi_prepare_and_post_rma("shmem putmem", ACLSHMEMI_OP_PUT, NO_NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, g_state_host.default_stream,
                                          g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("aclshmem_putmem failed");
    }
}

void aclshmem_getmem(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = aclshmemi_prepare_and_post_rma("shmem getmem", ACLSHMEMI_OP_GET, NO_NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, g_state_host.default_stream,
                                          g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("aclshmem_getmem failed");
    }
}

void aclshmem_putmem_nbi(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = aclshmemi_prepare_and_post_rma("aclshmem_putmem_nbi", ACLSHMEMI_OP_PUT, NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, g_state_host.default_stream,
                                          g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("aclshmem_putmem_nbi failed");
    }
}

void aclshmem_getmem_nbi(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = aclshmemi_prepare_and_post_rma("aclshmem_getmem_nbi", ACLSHMEMI_OP_GET, NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, g_state_host.default_stream,
                                          g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("aclshmem_getmem_nbi failed");
    }
}

void aclshmemx_getmem_on_stream(void* dst, void* src, size_t elem_size, int32_t pe, aclrtStream stream)
{
    if (stream == nullptr) {
        stream = g_state_host.default_stream;
    }
    int ret = aclshmemi_prepare_and_post_rma("aclshmemx_getmem_on_stream", ACLSHMEMI_OP_GET, NO_NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, stream,
                                          g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("aclshmemx_getmem_on_stream failed");
    }
}

void aclshmemx_putmem_on_stream(void* dst, void* src, size_t elem_size, int32_t pe, aclrtStream stream)
{
    if (stream == nullptr) {
        stream = g_state_host.default_stream;
    }
    int ret = aclshmemi_prepare_and_post_rma("aclshmemx_putmem_on_stream", ACLSHMEMI_OP_PUT, NO_NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, stream,
                                          g_state_host.default_block_num);
                                        
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemx_putmem_on_stream failed");
    }
}

void aclshmemx_quiet_on_stream(aclrtStream stream)
{
    if (stream == nullptr) {
        stream = g_state_host.default_stream;
    }
    int ret = aclshmemi_prepare_and_post_rma("aclshmemx_quiet_on_stream", ACLSHMEMI_OP_QUIET, NO_NBI,
                                          nullptr, nullptr, 0, 0, 0, nullptr, 0, 0, 1, 1,
                                          stream, g_state_host.default_block_num);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemx_quiet_on_stream failed");
    }
}
