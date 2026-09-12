/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

#include "shmem.h"
#include "unittest/utils/func_type.h"

constexpr uint64_t MESSAGE_SIZE = 64;

// ============================================================
// Atomic Add
// ============================================================
#define UDMA_ATOMIC_ADD_TEST_KERNEL(NAME, TYPE)                                                                        \
    extern "C" __global__ __aicore__ void test_udma_atomic_add_##NAME##_kernel(GM_ADDR gva, uint64_t config)           \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE value = 10;                                                                                               \
        aclshmemx_udma_atomic_add((__gm__ TYPE *)gva, value, peer);                                                    \
        aclshmemx_udma_quiet(peer);                                                                                    \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_ADD_FUNC_TYPE(UDMA_ATOMIC_ADD_TEST_KERNEL);

#define UDMA_ATOMIC_ADD_TEST(NAME, TYPE)                                                                               \
    void test_udma_atomic_add_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)             \
    {                                                                                                                  \
        test_udma_atomic_add_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                             \
    }
UDMA_ATOMIC_ADD_FUNC_TYPE(UDMA_ATOMIC_ADD_TEST);


// ============================================================
// Atomic Fetch Add
// ============================================================
#define UDMA_ATOMIC_FETCH_ADD_TEST_KERNEL(NAME, TYPE)                                                                  \
    extern "C" __global__ __aicore__ void test_udma_atomic_fetch_add_##NAME##_kernel(GM_ADDR gva, uint64_t config)     \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE value = 10;                                                                                               \
        TYPE return_value = aclshmemx_udma_atomic_fetch_add((__gm__ TYPE *)gva, value, peer);                          \
        aclshmemx_udma_quiet(peer);                                                                                    \
        *((__gm__ TYPE *)gva + 1) = return_value;                                                                      \
        dcci_cachelines((__gm__ uint8_t *)((__gm__ TYPE *)gva + 1), sizeof(TYPE));                                     \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_ADD_TEST_KERNEL);

#define UDMA_ATOMIC_FETCH_ADD_TEST(NAME, TYPE)                                                                         \
    void test_udma_atomic_fetch_add_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)       \
    {                                                                                                                  \
        test_udma_atomic_fetch_add_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                       \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_ADD_TEST);


// ============================================================
// Atomic Compare Swap
// ============================================================
#define UDMA_ATOMIC_COMPARE_SWAP_TEST_KERNEL(NAME, TYPE)                                                               \
    extern "C" __global__ __aicore__ void test_udma_atomic_compare_swap_##NAME##_kernel(GM_ADDR gva,uint64_t config)   \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE cond = 1;                                                                                                 \
        TYPE value = 10;                                                                                               \
        TYPE return_value = aclshmemx_udma_atomic_compare_swap((__gm__ TYPE *)gva, cond, value, peer);                 \
        aclshmemx_udma_quiet(peer);                                                                                    \
        *((__gm__ TYPE *)gva + 1) = return_value;                                                                      \
        dcci_cachelines((__gm__ uint8_t *)((__gm__ TYPE *)gva + 1), sizeof(TYPE));                                     \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_COMPARE_SWAP_TEST_KERNEL);

#define UDMA_ATOMIC_COMPARE_SWAP_TEST(NAME, TYPE)                                                                      \
    void test_udma_atomic_compare_swap_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)    \
    {                                                                                                                  \
        test_udma_atomic_compare_swap_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                    \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_COMPARE_SWAP_TEST);


// ============================================================
// Atomic Fetch
// ============================================================
#define UDMA_ATOMIC_FETCH_TEST_KERNEL(NAME, TYPE)                                                                      \
    extern "C" __global__ __aicore__ void test_udma_atomic_fetch_##NAME##_kernel(GM_ADDR gva, uint64_t config)         \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE return_value = aclshmemx_udma_atomic_fetch((__gm__ TYPE *)gva, peer);                                     \
        aclshmemx_udma_quiet(peer);                                                                                    \
        *((__gm__ TYPE *)gva + 1) = return_value;                                                                      \
        dcci_cachelines((__gm__ uint8_t *)((__gm__ TYPE *)gva + 1), sizeof(TYPE));                                     \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_TEST_KERNEL);

#define UDMA_ATOMIC_FETCH_TEST(NAME, TYPE)                                                                             \
    void test_udma_atomic_fetch_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)           \
    {                                                                                                                  \
        test_udma_atomic_fetch_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                           \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_TEST);


// ============================================================
// Atomic Set
// ============================================================
#define UDMA_ATOMIC_SET_TEST_KERNEL(NAME, TYPE)                                                                        \
    extern "C" __global__ __aicore__ void test_udma_atomic_set_##NAME##_kernel(GM_ADDR gva, uint64_t config)           \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE value = 100;                                                                                              \
        aclshmemx_udma_atomic_set((__gm__ TYPE *)gva, value, peer);                                                    \
        aclshmemx_udma_quiet(peer);                                                                                    \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_SET_TEST_KERNEL);

#define UDMA_ATOMIC_SET_TEST(NAME, TYPE)                                                                               \
    void test_udma_atomic_set_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)             \
    {                                                                                                                  \
        test_udma_atomic_set_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                             \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_SET_TEST);


// ============================================================
// Atomic Swap
// ============================================================
#define UDMA_ATOMIC_SWAP_TEST_KERNEL(NAME, TYPE)                                                                       \
    extern "C" __global__ __aicore__ void test_udma_atomic_swap_##NAME##_kernel(GM_ADDR gva, uint64_t config)          \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE value = 100;                                                                                              \
        TYPE return_value = aclshmemx_udma_atomic_swap((__gm__ TYPE *)gva, value, peer);                               \
        aclshmemx_udma_quiet(peer);                                                                                    \
        *((__gm__ TYPE *)gva + 1) = return_value;                                                                      \
        dcci_cachelines((__gm__ uint8_t *)((__gm__ TYPE *)gva + 1), sizeof(TYPE));                                     \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_SWAP_TEST_KERNEL);

#define UDMA_ATOMIC_SWAP_TEST(NAME, TYPE)                                                                              \
    void test_udma_atomic_swap_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)            \
    {                                                                                                                  \
        test_udma_atomic_swap_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                            \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_SWAP_TEST);


// ============================================================
// Atomic Fetch Inc
// ============================================================
#define UDMA_ATOMIC_FETCH_INC_TEST_KERNEL(NAME, TYPE)                                                                  \
    extern "C" __global__ __aicore__ void test_udma_atomic_fetch_inc_##NAME##_kernel(GM_ADDR gva, uint64_t config)     \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE return_value = aclshmemx_udma_atomic_fetch_inc((__gm__ TYPE *)gva, peer);                                 \
        aclshmemx_udma_quiet(peer);                                                                                    \
        *((__gm__ TYPE *)gva + 1) = return_value;                                                                      \
        dcci_cachelines((__gm__ uint8_t *)((__gm__ TYPE *)gva + 1), sizeof(TYPE));                                     \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_INC_TEST_KERNEL);

#define UDMA_ATOMIC_FETCH_INC_TEST(NAME, TYPE)                                                                         \
    void test_udma_atomic_fetch_inc_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)       \
    {                                                                                                                  \
        test_udma_atomic_fetch_inc_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                       \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_INC_TEST);


// ============================================================
// Atomic Inc
// ============================================================
#define UDMA_ATOMIC_INC_TEST_KERNEL(NAME, TYPE)                                                                        \
    extern "C" __global__ __aicore__ void test_udma_atomic_inc_##NAME##_kernel(GM_ADDR gva, uint64_t config)           \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        aclshmemx_udma_atomic_inc((__gm__ TYPE *)gva, peer);                                                           \
        aclshmemx_udma_quiet(peer);                                                                                    \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_INC_TEST_KERNEL);

#define UDMA_ATOMIC_INC_TEST(NAME, TYPE)                                                                               \
    void test_udma_atomic_inc_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)             \
    {                                                                                                                  \
        test_udma_atomic_inc_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                             \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_INC_TEST);


// ============================================================
// Atomic Fetch And
// ============================================================
#define UDMA_ATOMIC_FETCH_AND_TEST_KERNEL(NAME, TYPE)                                                                  \
    extern "C" __global__ __aicore__ void test_udma_atomic_fetch_and_##NAME##_kernel(GM_ADDR gva, uint64_t config)     \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE value = 0x0F;                                                                                             \
        TYPE return_value = aclshmemx_udma_atomic_fetch_and((__gm__ TYPE *)gva, value, peer);                          \
        aclshmemx_udma_quiet(peer);                                                                                    \
        *((__gm__ TYPE *)gva + 1) = return_value;                                                                      \
        dcci_cachelines((__gm__ uint8_t *)((__gm__ TYPE *)gva + 1), sizeof(TYPE));                                     \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_AND_TEST_KERNEL);

#define UDMA_ATOMIC_FETCH_AND_TEST(NAME, TYPE)                                                                         \
    void test_udma_atomic_fetch_and_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)       \
    {                                                                                                                  \
        test_udma_atomic_fetch_and_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                       \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_AND_TEST);


// ============================================================
// Atomic And
// ============================================================
#define UDMA_ATOMIC_AND_TEST_KERNEL(NAME, TYPE)                                                                        \
    extern "C" __global__ __aicore__ void test_udma_atomic_and_##NAME##_kernel(GM_ADDR gva, uint64_t config)           \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE value = 0x0F;                                                                                             \
        aclshmemx_udma_atomic_and((__gm__ TYPE *)gva, value, peer);                                                    \
        aclshmemx_udma_quiet(peer);                                                                                    \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_AND_TEST_KERNEL);

#define UDMA_ATOMIC_AND_TEST(NAME, TYPE)                                                                               \
    void test_udma_atomic_and_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)             \
    {                                                                                                                  \
        test_udma_atomic_and_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                             \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_AND_TEST);


// ============================================================
// Atomic Fetch Or
// ============================================================
#define UDMA_ATOMIC_FETCH_OR_TEST_KERNEL(NAME, TYPE)                                                                   \
    extern "C" __global__ __aicore__ void test_udma_atomic_fetch_or_##NAME##_kernel(GM_ADDR gva, uint64_t config)      \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE value = 0xF0;                                                                                             \
        TYPE return_value = aclshmemx_udma_atomic_fetch_or((__gm__ TYPE *)gva, value, peer);                           \
        aclshmemx_udma_quiet(peer);                                                                                    \
        *((__gm__ TYPE *)gva + 1) = return_value;                                                                      \
        dcci_cachelines((__gm__ uint8_t *)((__gm__ TYPE *)gva + 1), sizeof(TYPE));                                     \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_OR_TEST_KERNEL);

#define UDMA_ATOMIC_FETCH_OR_TEST(NAME, TYPE)                                                                          \
    void test_udma_atomic_fetch_or_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)        \
    {                                                                                                                  \
        test_udma_atomic_fetch_or_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_OR_TEST);


// ============================================================
// Atomic Or
// ============================================================
#define UDMA_ATOMIC_OR_TEST_KERNEL(NAME, TYPE)                                                                         \
    extern "C" __global__ __aicore__ void test_udma_atomic_or_##NAME##_kernel(GM_ADDR gva, uint64_t config)            \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE value = 0xF0;                                                                                             \
        aclshmemx_udma_atomic_or((__gm__ TYPE *)gva, value, peer);                                                     \
        aclshmemx_udma_quiet(peer);                                                                                    \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_OR_TEST_KERNEL);

#define UDMA_ATOMIC_OR_TEST(NAME, TYPE)                                                                                \
    void test_udma_atomic_or_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)              \
    {                                                                                                                  \
        test_udma_atomic_or_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                              \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_OR_TEST);


// ============================================================
// Atomic Fetch Xor
// ============================================================
#define UDMA_ATOMIC_FETCH_XOR_TEST_KERNEL(NAME, TYPE)                                                                  \
    extern "C" __global__ __aicore__ void test_udma_atomic_fetch_xor_##NAME##_kernel(GM_ADDR gva, uint64_t config)     \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE value = 0xFF;                                                                                             \
        TYPE return_value = aclshmemx_udma_atomic_fetch_xor((__gm__ TYPE *)gva, value, peer);                          \
        aclshmemx_udma_quiet(peer);                                                                                    \
        *((__gm__ TYPE *)gva + 1) = return_value;                                                                      \
        dcci_cachelines((__gm__ uint8_t *)((__gm__ TYPE *)gva + 1), sizeof(TYPE));                                     \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_XOR_TEST_KERNEL);

#define UDMA_ATOMIC_FETCH_XOR_TEST(NAME, TYPE)                                                                         \
    void test_udma_atomic_fetch_xor_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)       \
    {                                                                                                                  \
        test_udma_atomic_fetch_xor_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                       \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_FETCH_XOR_TEST);


// ============================================================
// Atomic Xor
// ============================================================
#define UDMA_ATOMIC_XOR_TEST_KERNEL(NAME, TYPE)                                                                        \
    extern "C" __global__ __aicore__ void test_udma_atomic_xor_##NAME##_kernel(GM_ADDR gva, uint64_t config)           \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int32_t rank = aclshmem_my_pe();                                                                               \
        int32_t rank_size = aclshmem_n_pes();                                                                          \
        int32_t peer = (rank + 1) % rank_size;                                                                         \
        TYPE value = 0xFF;                                                                                             \
        aclshmemx_udma_atomic_xor((__gm__ TYPE *)gva, value, peer);                                                    \
        aclshmemx_udma_quiet(peer);                                                                                    \
        aclshmem_barrier_all();                                                                                        \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_XOR_TEST_KERNEL);

#define UDMA_ATOMIC_XOR_TEST(NAME, TYPE)                                                                               \
    void test_udma_atomic_xor_##NAME##_do(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config)             \
    {                                                                                                                  \
        test_udma_atomic_xor_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);                             \
    }
UDMA_ATOMIC_FUNC_TYPE(UDMA_ATOMIC_XOR_TEST);
