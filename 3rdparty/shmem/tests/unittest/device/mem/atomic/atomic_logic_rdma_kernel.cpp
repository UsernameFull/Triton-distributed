/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file atomic_logic_rdma_kernel.cpp
 * @brief Unit test kernels for RDMA atomic logic operations:
 *        - atomic_and
 *        - atomic_or
 *        - atomic_xor
 *        - atomic_fetch_and
 *        - atomic_fetch_or
 *        - atomic_fetch_xor
 */

#include "kernel_operator.h"
#include "shmem.h"
#include "unittest/utils/func_type.h"

constexpr uint64_t MESSAGE_SIZE = 64;

/*****************************************************************************
 *                    atomic_and test (RDMA)                                  *
 *****************************************************************************/
#ifdef ACLSHMEMI_RDMA_K_BACKEND_XSCALE
/**
 * @brief RDMA atomic_and kernel test
 *        Each PE clears its own bit in other PEs' slots
 */
#define RDMA_ATOMIC_AND_TEST_KERNEL(NAME, TYPE)                                                                   \
    extern "C" __global__ __aicore__ void test_rdma_atomic_and_##NAME##_kernel(__gm__ TYPE* gva, uint64_t config) \
    {                                                                                                             \
        util_set_ffts_config(config);                                                                             \
        int64_t pe = aclshmem_my_pe();                                                                            \
        int64_t pe_size = aclshmem_n_pes();                                                                       \
        __gm__ TYPE* dst_addr;                                                                                    \
                                                                                                                  \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                                          \
            if (peer == pe)                                                                                       \
                continue;                                                                                         \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                                    \
            if (g_coreType == AscendC::AIV && AscendC::GetSubBlockIdx() == 0) {                                   \
                aclshmemx_roce_atomic_and((__gm__ TYPE*)dst_addr, ~(TYPE)(1LLU << pe), peer);                     \
            }                                                                                                     \
        }                                                                                                         \
        aclshmem_barrier_all();                                                                                   \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_AND_TEST_KERNEL);

#define RDMA_ATOMIC_AND_TEST(NAME, TYPE)                                                                \
    void test_rdma_atomic_and_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config) \
    {                                                                                                   \
        test_rdma_atomic_and_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);              \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_AND_TEST);

/*****************************************************************************
 *                    atomic_fetch_and test (RDMA)                            *
 *****************************************************************************/
/**
 * @brief RDMA atomic_fetch_and kernel test
 *        Each PE clears its own bit in other PEs' slots and returns old value
 *        Verifies return value has own bit still set (not yet cleared)
 */
#define RDMA_ATOMIC_FETCH_AND_TEST_KERNEL(NAME, TYPE)                                                         \
    extern "C" __global__ __aicore__ void test_rdma_atomic_fetch_and_##NAME##_kernel(                         \
        __gm__ TYPE* gva, __gm__ int* error_flag, uint64_t pe_size, uint64_t config)                          \
    {                                                                                                         \
        util_set_ffts_config(config);                                                                         \
        int64_t pe = aclshmem_my_pe();                                                                        \
        int64_t pe_size_val = aclshmem_n_pes();                                                               \
        __gm__ TYPE* dst_addr;                                                                                \
        TYPE old_val;                                                                                         \
                                                                                                              \
        for (int64_t peer = 0; peer < pe_size_val; peer++) {                                                  \
            if (peer == pe)                                                                                   \
                continue;                                                                                     \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                                \
            if (g_coreType == AscendC::AIV && AscendC::GetSubBlockIdx() == 0) {                               \
                old_val = aclshmemx_roce_atomic_fetch_and((__gm__ TYPE*)dst_addr, ~(TYPE)(1LLU << pe), peer); \
                if ((old_val & ((TYPE)(1LLU << pe))) == 0) {                                                  \
                    *error_flag = 1;                                                                          \
                }                                                                                             \
            }                                                                                                 \
        }                                                                                                     \
        aclshmem_barrier_all();                                                                               \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_FETCH_AND_TEST_KERNEL);

#define RDMA_ATOMIC_FETCH_AND_TEST(NAME, TYPE)                                                                        \
    void test_rdma_atomic_fetch_and_##NAME##_do(                                                                      \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t pe_size, uint64_t config)              \
    {                                                                                                                 \
        test_rdma_atomic_fetch_and_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, error_flag, pe_size, config); \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_FETCH_AND_TEST);

/*****************************************************************************
 *                    atomic_or test (RDMA)                                   *
 *****************************************************************************/
/**
 * @brief RDMA atomic_or kernel test
 *        Each PE sets its own bit in other PEs' slots
 */
#define RDMA_ATOMIC_OR_TEST_KERNEL(NAME, TYPE)                                                                   \
    extern "C" __global__ __aicore__ void test_rdma_atomic_or_##NAME##_kernel(__gm__ TYPE* gva, uint64_t config) \
    {                                                                                                            \
        util_set_ffts_config(config);                                                                            \
        int64_t pe = aclshmem_my_pe();                                                                           \
        int64_t pe_size = aclshmem_n_pes();                                                                      \
        __gm__ TYPE* dst_addr;                                                                                   \
                                                                                                                 \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                                         \
            if (peer == pe)                                                                                      \
                continue;                                                                                        \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                                   \
            if (g_coreType == AscendC::AIV && AscendC::GetSubBlockIdx() == 0) {                                  \
                aclshmemx_roce_atomic_or((__gm__ TYPE*)dst_addr, (TYPE)(1LLU << pe), peer);                      \
            }                                                                                                    \
        }                                                                                                        \
        aclshmem_barrier_all();                                                                                  \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_OR_TEST_KERNEL);

#define RDMA_ATOMIC_OR_TEST(NAME, TYPE)                                                                \
    void test_rdma_atomic_or_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config) \
    {                                                                                                  \
        test_rdma_atomic_or_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);              \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_OR_TEST);

/*****************************************************************************
 *                    atomic_fetch_or test (RDMA)                             *
 *****************************************************************************/
/**
 * @brief RDMA atomic_fetch_or kernel test
 *        Each PE sets its own bit in other PEs' slots and returns old value
 *        Verifies return value has own bit not yet set
 */
#define RDMA_ATOMIC_FETCH_OR_TEST_KERNEL(NAME, TYPE)                                                        \
    extern "C" __global__ __aicore__ void test_rdma_atomic_fetch_or_##NAME##_kernel(                        \
        __gm__ TYPE* gva, __gm__ int* error_flag, uint64_t pe_size, uint64_t config)                        \
    {                                                                                                       \
        util_set_ffts_config(config);                                                                       \
        int64_t pe = aclshmem_my_pe();                                                                      \
        int64_t pe_size_val = aclshmem_n_pes();                                                             \
        __gm__ TYPE* dst_addr;                                                                              \
        TYPE old_val;                                                                                       \
                                                                                                            \
        for (int64_t peer = 0; peer < pe_size_val; peer++) {                                                \
            if (peer == pe)                                                                                 \
                continue;                                                                                   \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                              \
            if (g_coreType == AscendC::AIV && AscendC::GetSubBlockIdx() == 0) {                             \
                old_val = aclshmemx_roce_atomic_fetch_or((__gm__ TYPE*)dst_addr, (TYPE)(1LLU << pe), peer); \
                if ((old_val & ((TYPE)(1LLU << pe))) != 0) {                                                \
                    *error_flag = 1;                                                                        \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
        aclshmem_barrier_all();                                                                             \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_FETCH_OR_TEST_KERNEL);

#define RDMA_ATOMIC_FETCH_OR_TEST(NAME, TYPE)                                                                        \
    void test_rdma_atomic_fetch_or_##NAME##_do(                                                                      \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t pe_size, uint64_t config)             \
    {                                                                                                                \
        test_rdma_atomic_fetch_or_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, error_flag, pe_size, config); \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_FETCH_OR_TEST);

/*****************************************************************************
 *                    atomic_xor test (RDMA)                                  *
 *****************************************************************************/
/**
 * @brief RDMA atomic_xor kernel test
 *        Each PE toggles its own bit in other PEs' slots (1→0)
 *        Initial: all bits = 1, final: lower pe_size bits = 0
 */
#define RDMA_ATOMIC_XOR_TEST_KERNEL(NAME, TYPE)                                                                   \
    extern "C" __global__ __aicore__ void test_rdma_atomic_xor_##NAME##_kernel(__gm__ TYPE* gva, uint64_t config) \
    {                                                                                                             \
        util_set_ffts_config(config);                                                                             \
        int64_t pe = aclshmem_my_pe();                                                                            \
        int64_t pe_size = aclshmem_n_pes();                                                                       \
        __gm__ TYPE* dst_addr;                                                                                    \
                                                                                                                  \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                                          \
            if (peer == pe)                                                                                       \
                continue;                                                                                         \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                                    \
            if (g_coreType == AscendC::AIV && AscendC::GetSubBlockIdx() == 0) {                                   \
                aclshmemx_roce_atomic_xor((__gm__ TYPE*)dst_addr, (TYPE)(1LLU << pe), peer);                      \
            }                                                                                                     \
        }                                                                                                         \
        aclshmem_barrier_all();                                                                                   \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_XOR_TEST_KERNEL);

#define RDMA_ATOMIC_XOR_TEST(NAME, TYPE)                                                                \
    void test_rdma_atomic_xor_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config) \
    {                                                                                                   \
        test_rdma_atomic_xor_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);              \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_XOR_TEST);

/*****************************************************************************
 *                    atomic_fetch_xor test (RDMA)                            *
 *****************************************************************************/
/**
 * @brief RDMA atomic_fetch_xor kernel test
 *        Each PE toggles its own bit in other PEs' slots and returns old value
 *        Verifies return value has own bit still set (1)
 */
#define RDMA_ATOMIC_FETCH_XOR_TEST_KERNEL(NAME, TYPE)                                                        \
    extern "C" __global__ __aicore__ void test_rdma_atomic_fetch_xor_##NAME##_kernel(                        \
        __gm__ TYPE* gva, __gm__ int* error_flag, uint64_t pe_size, uint64_t config)                         \
    {                                                                                                        \
        util_set_ffts_config(config);                                                                        \
        int64_t pe = aclshmem_my_pe();                                                                       \
        int64_t pe_size_val = aclshmem_n_pes();                                                              \
        __gm__ TYPE* dst_addr;                                                                               \
        TYPE old_val;                                                                                        \
                                                                                                             \
        for (int64_t peer = 0; peer < pe_size_val; peer++) {                                                 \
            if (peer == pe)                                                                                  \
                continue;                                                                                    \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                               \
            if (g_coreType == AscendC::AIV && AscendC::GetSubBlockIdx() == 0) {                              \
                old_val = aclshmemx_roce_atomic_fetch_xor((__gm__ TYPE*)dst_addr, (TYPE)(1LLU << pe), peer); \
                if ((old_val & ((TYPE)(1LLU << pe))) == 0) {                                                 \
                    *error_flag = 1;                                                                         \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
        aclshmem_barrier_all();                                                                              \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_FETCH_XOR_TEST_KERNEL);

#define RDMA_ATOMIC_FETCH_XOR_TEST(NAME, TYPE)                                                                        \
    void test_rdma_atomic_fetch_xor_##NAME##_do(                                                                      \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t pe_size, uint64_t config)              \
    {                                                                                                                 \
        test_rdma_atomic_fetch_xor_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, error_flag, pe_size, config); \
    }
ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(RDMA_ATOMIC_FETCH_XOR_TEST);
#endif // ACLSHMEMI_RDMA_K_BACKEND_XSCALE
