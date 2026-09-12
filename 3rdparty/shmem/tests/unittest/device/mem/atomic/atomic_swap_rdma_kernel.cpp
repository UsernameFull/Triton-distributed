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
 * @file atomic_swap_rdma_kernel.cpp
 * @brief Unit test kernels for RDMA atomic swap operations:
 *        - atomic_fetch
 *        - atomic_set
 *        - atomic_swap
 *        - atomic_compare_swap
 */

#include "kernel_operator.h"
#include "shmem.h"
#include "unittest/utils/func_type.h"

constexpr uint64_t MESSAGE_SIZE = 64;

/*****************************************************************************
 *                    atomic_fetch test (RDMA)                                *
 *****************************************************************************/
#ifdef ACLSHMEMI_RDMA_K_BACKEND_XSCALE
/**
 * @brief RDMA atomic_fetch kernel test
 *        Each PE reads from other PEs' memory and verifies return value
 */
#define RDMA_ATOMIC_FETCH_TEST_KERNEL(NAME, TYPE)                                      \
    extern "C" __global__ __aicore__ void test_rdma_atomic_fetch_##NAME##_kernel(      \
        __gm__ TYPE* gva, __gm__ int* error_flag, uint64_t config)                     \
    {                                                                                  \
        util_set_ffts_config(config);                                                  \
        int64_t pe = aclshmem_my_pe();                                                 \
        int64_t pe_size = aclshmem_n_pes();                                            \
        __gm__ TYPE* dst_addr;                                                         \
        TYPE fetch_val;                                                                \
                                                                                       \
        for (int64_t peer = 0; peer < pe_size; peer++) {                               \
            if (peer == pe)                                                            \
                continue;                                                              \
            dst_addr = gva + peer * MESSAGE_SIZE / sizeof(TYPE);                       \
            if (g_coreType == AscendC::AIV && AscendC::GetSubBlockIdx() == 0) {        \
                fetch_val = aclshmemx_roce_atomic_fetch((__gm__ TYPE*)dst_addr, peer); \
                if (fetch_val != peer + 1) {                                           \
                    *error_flag = 1;                                                   \
                }                                                                      \
            }                                                                          \
        }                                                                              \
        aclshmem_barrier_all();                                                        \
    }
ACLSHMEM_RDMA_ATOMIC_SWAP_FUNC_TYPE(RDMA_ATOMIC_FETCH_TEST_KERNEL);

#define RDMA_ATOMIC_FETCH_TEST(NAME, TYPE)                                                               \
    void test_rdma_atomic_fetch_##NAME##_do(                                                             \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t config)                   \
    {                                                                                                    \
        test_rdma_atomic_fetch_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, error_flag, config); \
    }
ACLSHMEM_RDMA_ATOMIC_SWAP_FUNC_TYPE(RDMA_ATOMIC_FETCH_TEST);

/*****************************************************************************
 *                    atomic_set test (RDMA)                                  *
 *****************************************************************************/
/**
 * @brief RDMA atomic_set kernel test
 *        Each PE asks other PEs to set its own slot to 1
 */
#define RDMA_ATOMIC_SET_TEST_KERNEL(NAME, TYPE)                                                                   \
    extern "C" __global__ __aicore__ void test_rdma_atomic_set_##NAME##_kernel(__gm__ TYPE* gva, uint64_t config) \
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
                aclshmemx_roce_atomic_set((__gm__ TYPE*)dst_addr, static_cast<TYPE>(1), peer);                    \
            }                                                                                                     \
        }                                                                                                         \
        aclshmem_barrier_all();                                                                                   \
    }
ACLSHMEM_RDMA_ATOMIC_SWAP_FUNC_TYPE(RDMA_ATOMIC_SET_TEST_KERNEL);

#define RDMA_ATOMIC_SET_TEST(NAME, TYPE)                                                                \
    void test_rdma_atomic_set_##NAME##_do(uint32_t block_dim, void* stream, TYPE* gva, uint64_t config) \
    {                                                                                                   \
        test_rdma_atomic_set_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, config);              \
    }
ACLSHMEM_RDMA_ATOMIC_SWAP_FUNC_TYPE(RDMA_ATOMIC_SET_TEST);

/*****************************************************************************
 *                    atomic_swap test (RDMA)                                 *
 *****************************************************************************/
/**
 * @brief RDMA atomic_swap kernel test
 *        Each PE asks other PEs to swap its own slot to 1
 */
#define RDMA_ATOMIC_SWAP_TEST_KERNEL(NAME, TYPE)                                                           \
    extern "C" __global__ __aicore__ void test_rdma_atomic_swap_##NAME##_kernel(                           \
        __gm__ TYPE* gva, __gm__ int* error_flag, uint64_t config)                                         \
    {                                                                                                      \
        util_set_ffts_config(config);                                                                      \
        int64_t pe = aclshmem_my_pe();                                                                     \
        int64_t pe_size = aclshmem_n_pes();                                                                \
        __gm__ TYPE* dst_addr;                                                                             \
        TYPE swap_val;                                                                                     \
                                                                                                           \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                                   \
            if (peer == pe)                                                                                \
                continue;                                                                                  \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                             \
            if (g_coreType == AscendC::AIV && AscendC::GetSubBlockIdx() == 0) {                            \
                swap_val = aclshmemx_roce_atomic_swap((__gm__ TYPE*)dst_addr, static_cast<TYPE>(1), peer); \
                if (swap_val != 0) {                                                                       \
                    *error_flag = 1;                                                                       \
                }                                                                                          \
            }                                                                                              \
        }                                                                                                  \
        aclshmem_barrier_all();                                                                            \
    }
ACLSHMEM_RDMA_ATOMIC_SWAP_FUNC_TYPE(RDMA_ATOMIC_SWAP_TEST_KERNEL);

#define RDMA_ATOMIC_SWAP_TEST(NAME, TYPE)                                                               \
    void test_rdma_atomic_swap_##NAME##_do(                                                             \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t config)                  \
    {                                                                                                   \
        test_rdma_atomic_swap_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, error_flag, config); \
    }
ACLSHMEM_RDMA_ATOMIC_SWAP_FUNC_TYPE(RDMA_ATOMIC_SWAP_TEST);

/*****************************************************************************
 *                    atomic_compare_swap test (RDMA)                         *
 *****************************************************************************/
/**
 * @brief RDMA atomic_compare_swap kernel test
 *        Each PE asks other PEs to compare and swap its own slot
 */
#define RDMA_ATOMIC_COMPARE_SWAP_TEST_KERNEL(NAME, TYPE)                                       \
    extern "C" __global__ __aicore__ void test_rdma_atomic_compare_swap_##NAME##_kernel(       \
        __gm__ TYPE* gva, __gm__ int* error_flag, uint64_t config)                             \
    {                                                                                          \
        util_set_ffts_config(config);                                                          \
        int64_t pe = aclshmem_my_pe();                                                         \
        int64_t pe_size = aclshmem_n_pes();                                                    \
        __gm__ TYPE* dst_addr;                                                                 \
        TYPE swap_val;                                                                         \
                                                                                               \
        for (int64_t peer = 0; peer < pe_size; peer++) {                                       \
            if (peer == pe)                                                                    \
                continue;                                                                      \
            dst_addr = gva + pe * MESSAGE_SIZE / sizeof(TYPE);                                 \
            if (g_coreType == AscendC::AIV && AscendC::GetSubBlockIdx() == 0) {                \
                swap_val = aclshmemx_roce_atomic_compare_swap(                                 \
                    (__gm__ TYPE*)dst_addr, static_cast<TYPE>(0), static_cast<TYPE>(1), peer); \
                if (swap_val != 0) {                                                           \
                    *error_flag = 1;                                                           \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
        aclshmem_barrier_all();                                                                \
    }
ACLSHMEM_RDMA_ATOMIC_CAS_FUNC_TYPE(RDMA_ATOMIC_COMPARE_SWAP_TEST_KERNEL);

#define RDMA_ATOMIC_COMPARE_SWAP_TEST(NAME, TYPE)                                                               \
    void test_rdma_atomic_compare_swap_##NAME##_do(                                                             \
        uint32_t block_dim, void* stream, TYPE* gva, int* error_flag, uint64_t config)                          \
    {                                                                                                           \
        test_rdma_atomic_compare_swap_##NAME##_kernel<<<block_dim, nullptr, stream>>>(gva, error_flag, config); \
    }
ACLSHMEM_RDMA_ATOMIC_CAS_FUNC_TYPE(RDMA_ATOMIC_COMPARE_SWAP_TEST);
#endif // ACLSHMEMI_RDMA_K_BACKEND_XSCALE
