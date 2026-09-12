/**
?* Copyright (c) 2026 Huawei Technologies Co., Ltd.
?* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
?* CANN Open Software License Agreement Version 2.0 (the "License").
?* Please refer to the License for details. You may not use this file except in compliance with the License.
?* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
?* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
?* See LICENSE in the root of the software repository for the full text of the License.
?*/
#include "kernel_operator.h"

#include "shmem.h"
#include "unittest/utils/func_type.h"

#define IPUT_MEM(NAME, TYPE)                                                                                          \
__global__ __aicore__ void aclshmem_##NAME##_iput_kernel(GM_ADDR dest, GM_ADDR source, ptrdiff_t dst, ptrdiff_t sst,  \
                                                          size_t nelems, int pe)                                      \
{                                                                                                                     \
    aclshmem_##NAME##_iput(reinterpret_cast<__gm__ TYPE *>(dest), reinterpret_cast<__gm__ TYPE *>(source), dst, sst,  \
                           nelems, pe);                                                                               \
}                                                                                                                     \
                                                                                                                      \
void aclshmem_##NAME##_iput_wrapper(uint32_t block_dim, void *stream, TYPE *dest, TYPE *source,                       \
                                    ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {                            \
    aclshmem_##NAME##_iput_kernel<<<block_dim, nullptr, stream>>>(reinterpret_cast<uint8_t *>(dest),                  \
        reinterpret_cast<uint8_t *>(source), dst, sst, nelems, pe);                                                   \
}

ACLSHMEM_MEM_PUT_GET_FUNC(IPUT_MEM)
#undef IPUT_MEM


#define IGET_MEM(NAME, TYPE)                                                                                          \
__global__ __aicore__ void aclshmem_##NAME##_iget_kernel(GM_ADDR dest, GM_ADDR source, ptrdiff_t dst, ptrdiff_t sst,  \
                                                          size_t nelems, int pe)                                      \
{                                                                                                                     \
    aclshmem_##NAME##_iget(reinterpret_cast<__gm__ TYPE *>(dest), reinterpret_cast<__gm__ TYPE *>(source), dst, sst,  \
                           nelems, pe);                                                                               \
}                                                                                                                     \
                                                                                                                      \
void aclshmem_##NAME##_iget_wrapper(uint32_t block_dim, void *stream, TYPE *dest, TYPE *source,                       \
                                    ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {                            \
    aclshmem_##NAME##_iget_kernel<<<block_dim, nullptr, stream>>>(reinterpret_cast<uint8_t *>(dest),                  \
        reinterpret_cast<uint8_t *>(source), dst, sst, nelems, pe);                                                   \
}

ACLSHMEM_MEM_PUT_GET_FUNC(IGET_MEM)
#undef IGET_MEM


#define IPUT_SIZE(BITS, TYPE)                                                                                        \
__global__ __aicore__ void aclshmem_iput##BITS##_kernel(GM_ADDR dest, GM_ADDR source, ptrdiff_t dst, ptrdiff_t sst,  \
                                                          size_t nelems, int pe)                                     \
{                                                                                                                    \
    aclshmem_iput##BITS(dest, source, dst, sst, nelems, pe);                                                         \
}                                                                                                                    \
                                                                                                                     \
void aclshmem_iput##BITS##_wrapper(uint32_t block_dim, void *stream, TYPE *dest, TYPE *source,                       \
                                    ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {                           \
    aclshmem_iput##BITS##_kernel<<<block_dim, nullptr, stream>>>(reinterpret_cast<uint8_t *>(dest),                  \
        reinterpret_cast<uint8_t *>(source), dst, sst, nelems, pe);                                                  \
}

ACLSHMEM_PUT_GET_SIZE_FUNC(IPUT_SIZE)
#undef IPUT_SIZE


#define IGET_SIZE(BITS, TYPE)                                                                                        \
__global__ __aicore__ void aclshmem_iget##BITS##_kernel(GM_ADDR dest, GM_ADDR source, ptrdiff_t dst, ptrdiff_t sst,  \
                                                          size_t nelems, int pe)                                     \
{                                                                                                                    \
    aclshmem_iget##BITS(dest, source, dst, sst, nelems, pe);                                                         \
}                                                                                                                    \
                                                                                                                     \
void aclshmem_iget##BITS##_wrapper(uint32_t block_dim, void *stream, TYPE *dest, TYPE *source,                       \
                                    ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {                           \
    aclshmem_iget##BITS##_kernel<<<block_dim, nullptr, stream>>>(reinterpret_cast<uint8_t *>(dest),                  \
        reinterpret_cast<uint8_t *>(source), dst, sst, nelems, pe);                                                  \
}

ACLSHMEM_PUT_GET_SIZE_FUNC(IGET_SIZE)
#undef IGET_SIZE