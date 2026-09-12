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
 * @file highlevel_mem_rdma_test_kernel.cpp
 * @brief Unit tests for high-level RMA interfaces under RDMA/ROCE engine
 *
 * This file contains device-side kernel implementations for testing:
 * - aclshmem_getmem: blocking synchronous GET operation
 * - aclshmem_putmem: blocking synchronous PUT operation
 * - aclshmem_getmem_nbi: non-blocking GET operation
 * - aclshmem_putmem_nbi: non-blocking PUT operation
 *
 * Multi-type kernels for typed PUT/GET (using ACLSHMEM_MEM_PUT_GET_FUNC macro):
 * - aclshmem_##NAME##_put + aclshmem_##NAME##_get: typed blocking PUT+GET for all 11 types
 * - aclshmem_##NAME##_put_nbi + aclshmem_##NAME##_get_nbi: typed non-blocking PUT+GET for all 11 types
 * - aclshmem_##NAME##_put_nbi + aclshmem_##NAME##_get_nbi (GlobalTensor): typed non-blocking PUT+GET with GlobalTensor
 * for all 11 types
 * - Supported types: float, double, int8, int16, int32, int64, uint8, uint16, uint32, uint64, char
 */

#include "kernel_operator.h"
#include "shmem.h"

constexpr uint64_t MESSAGE_SIZE = 64;

/**
 * Test aclshmem_getmem (blocking/synchronous GET) for RDMA/ROCE engine
 * Each PE gets data from all other peers using aclshmem_getmem
 */
extern "C" __global__ __aicore__ void RDMAGetmemTest(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;

    if ASCEND_IS_AIV {
        int64_t my_pe = aclshmem_my_pe();
        int64_t n_pes = aclshmem_n_pes();

        for (int64_t peer = 0; peer < n_pes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            // Get data from peer: read from peer's region, write to my region
            GM_ADDR src_addr = gva + peer * MESSAGE_SIZE;
            GM_ADDR dst_addr = gva + peer * MESSAGE_SIZE;
            aclshmem_getmem(dst_addr, src_addr, MESSAGE_SIZE, peer);
        }
    }
}

void test_rdma_getmem(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAGetmemTest<<<block_dim, nullptr, stream>>>(gva, config);
}

/**
 * Test aclshmem_putmem (blocking/synchronous PUT) for RDMA/ROCE engine
 * Each PE puts data to all other peers using aclshmem_putmem
 */
extern "C" __global__ __aicore__ void RDMAPutmemTest(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;

    if ASCEND_IS_AIV {
        int64_t my_pe = aclshmem_my_pe();
        int64_t n_pes = aclshmem_n_pes();

        for (int64_t peer = 0; peer < n_pes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            // Put data to peer: read from my region, write to peer's region
            GM_ADDR src_addr = gva + my_pe * MESSAGE_SIZE;
            GM_ADDR dst_addr = gva + my_pe * MESSAGE_SIZE;
            aclshmem_putmem(dst_addr, src_addr, MESSAGE_SIZE, peer);
        }
    }
}

void test_rdma_putmem(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAPutmemTest<<<block_dim, nullptr, stream>>>(gva, config);
}

/**
 * Test aclshmem_getmem_nbi (non-blocking GET) for RDMA/ROCE engine
 * Each PE gets data from all other peers using aclshmem_getmem_nbi
 */
extern "C" __global__ __aicore__ void RDMAGetmemNbiTest(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;

    if ASCEND_IS_AIV {
        int64_t my_pe = aclshmem_my_pe();
        int64_t n_pes = aclshmem_n_pes();

        for (int64_t peer = 0; peer < n_pes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            // Get data from peer: read from peer's region, write to my region
            GM_ADDR src_addr = gva + peer * MESSAGE_SIZE;
            GM_ADDR dst_addr = gva + peer * MESSAGE_SIZE;
            aclshmem_getmem_nbi(dst_addr, src_addr, MESSAGE_SIZE, peer);
            // Wait for each non-blocking GET operation to complete
            aclshmem_quiet();
        }
    }
}

void test_rdma_getmem_nbi(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAGetmemNbiTest<<<block_dim, nullptr, stream>>>(gva, config);
}

/**
 * Test aclshmem_putmem_nbi (non-blocking PUT) for RDMA/ROCE engine
 * Each PE puts data to all other peers using aclshmem_putmem_nbi
 */
extern "C" __global__ __aicore__ void RDMAPutmemNbiTest(GM_ADDR gva, uint64_t config)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;

    if ASCEND_IS_AIV {
        int64_t my_pe = aclshmem_my_pe();
        int64_t n_pes = aclshmem_n_pes();

        for (int64_t peer = 0; peer < n_pes; peer++) {
            if (peer == my_pe) {
                continue;
            }
            // Put data to peer: read from my region, write to peer's region
            GM_ADDR src_addr = gva + my_pe * MESSAGE_SIZE;
            GM_ADDR dst_addr = gva + my_pe * MESSAGE_SIZE;
            aclshmem_putmem_nbi(dst_addr, src_addr, MESSAGE_SIZE, peer);
            // Wait for each non-blocking PUT operation to complete
            aclshmem_quiet();
        }
    }
}

void test_rdma_putmem_nbi(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)
{
    RDMAPutmemNbiTest<<<block_dim, nullptr, stream>>>(gva, config);
}

// ============================================================================
// Section: Multi-type typed PUT/GET kernels (using ACLSHMEM_MEM_PUT_GET_FUNC macro)
// ============================================================================

#include "unittest/utils/func_type.h"

/**
 * Macro to generate kernel and wrapper for each typed PUT/GET type
 * Generates:
 * - RDMA_##NAME##PutTestKernel: Device kernel for typed blocking PUT
 * - RDMA_##NAME##GetTestKernel: Device kernel for typed blocking GET
 * - test_rdma_##NAME##_put: Host wrapper for PUT kernel
 * - test_rdma_##NAME##_get: Host wrapper for GET kernel
 */
#define GEN_PUT_GET_KERNEL(NAME, TYPE)                                                                  \
    extern "C" __global__ __aicore__ void RDMA_##NAME##PutTestKernel(GM_ADDR gva, uint64_t config)      \
    {                                                                                                   \
        util_set_ffts_config(config);                                                                   \
        AscendC::TPipe pipe;                                                                            \
                                                                                                        \
        if ASCEND_IS_AIV {                                                                              \
            int64_t my_pe = aclshmem_my_pe();                                                           \
            int64_t n_pes = aclshmem_n_pes();                                                           \
                                                                                                        \
            for (int64_t peer = 0; peer < n_pes; peer++) {                                              \
                if (peer == my_pe) {                                                                    \
                    continue;                                                                           \
                }                                                                                       \
                GM_ADDR src_addr = gva + my_pe * MESSAGE_SIZE;                                          \
                GM_ADDR dst_addr = gva + my_pe * MESSAGE_SIZE;                                          \
                aclshmem_##NAME##_put(                                                                  \
                    reinterpret_cast<__gm__ TYPE*>(dst_addr), reinterpret_cast<__gm__ TYPE*>(src_addr), \
                    MESSAGE_SIZE / sizeof(TYPE), peer);                                                 \
            }                                                                                           \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    extern "C" __global__ __aicore__ void RDMA_##NAME##GetTestKernel(GM_ADDR gva, uint64_t config)      \
    {                                                                                                   \
        util_set_ffts_config(config);                                                                   \
        AscendC::TPipe pipe;                                                                            \
                                                                                                        \
        if ASCEND_IS_AIV {                                                                              \
            int64_t my_pe = aclshmem_my_pe();                                                           \
            int64_t n_pes = aclshmem_n_pes();                                                           \
                                                                                                        \
            for (int64_t peer = 0; peer < n_pes; peer++) {                                              \
                if (peer == my_pe) {                                                                    \
                    continue;                                                                           \
                }                                                                                       \
                GM_ADDR src_addr = gva + peer * MESSAGE_SIZE;                                           \
                GM_ADDR dst_addr = gva + peer * MESSAGE_SIZE;                                           \
                aclshmem_##NAME##_get(                                                                  \
                    reinterpret_cast<__gm__ TYPE*>(dst_addr), reinterpret_cast<__gm__ TYPE*>(src_addr), \
                    MESSAGE_SIZE / sizeof(TYPE), peer);                                                 \
            }                                                                                           \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    void test_rdma_##NAME##_put(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)        \
    {                                                                                                   \
        RDMA_##NAME##PutTestKernel<<<block_dim, nullptr, stream>>>(gva, config);                        \
    }                                                                                                   \
                                                                                                        \
    void test_rdma_##NAME##_get(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)        \
    {                                                                                                   \
        RDMA_##NAME##GetTestKernel<<<block_dim, nullptr, stream>>>(gva, config);                        \
    }

// Generate kernels for all supported types
ACLSHMEM_MEM_PUT_GET_FUNC(GEN_PUT_GET_KERNEL);
#undef GEN_PUT_GET_KERNEL

// ============================================================================
// Section: Multi-type typed PUT_NBI/GET_NBI kernels (using ACLSHMEM_MEM_PUT_GET_FUNC macro)
// ============================================================================

/**
 * Macro to generate kernel and wrapper for each typed PUT_NBI/GET_NBI type
 * Generates:
 * - RDMA_##NAME##PutNbiTestKernel: Device kernel for typed non-blocking PUT
 * - RDMA_##NAME##GetNbiTestKernel: Device kernel for typed non-blocking GET
 * - test_rdma_##NAME##_put_nbi: Host wrapper for PUT_NBI kernel
 * - test_rdma_##NAME##_get_nbi: Host wrapper for GET_NBI kernel
 */
#define GEN_PUT_GET_NBI_KERNEL(NAME, TYPE)                                                              \
    extern "C" __global__ __aicore__ void RDMA_##NAME##PutNbiTestKernel(GM_ADDR gva, uint64_t config)   \
    {                                                                                                   \
        util_set_ffts_config(config);                                                                   \
        AscendC::TPipe pipe;                                                                            \
                                                                                                        \
        if ASCEND_IS_AIV {                                                                              \
            int64_t my_pe = aclshmem_my_pe();                                                           \
            int64_t n_pes = aclshmem_n_pes();                                                           \
                                                                                                        \
            for (int64_t peer = 0; peer < n_pes; peer++) {                                              \
                if (peer == my_pe) {                                                                    \
                    continue;                                                                           \
                }                                                                                       \
                GM_ADDR src_addr = gva + my_pe * MESSAGE_SIZE;                                          \
                GM_ADDR dst_addr = gva + my_pe * MESSAGE_SIZE;                                          \
                aclshmem_##NAME##_put_nbi(                                                              \
                    reinterpret_cast<__gm__ TYPE*>(dst_addr), reinterpret_cast<__gm__ TYPE*>(src_addr), \
                    MESSAGE_SIZE / sizeof(TYPE), peer);                                                 \
                aclshmem_quiet();                                                                       \
            }                                                                                           \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    extern "C" __global__ __aicore__ void RDMA_##NAME##GetNbiTestKernel(GM_ADDR gva, uint64_t config)   \
    {                                                                                                   \
        util_set_ffts_config(config);                                                                   \
        AscendC::TPipe pipe;                                                                            \
                                                                                                        \
        if ASCEND_IS_AIV {                                                                              \
            int64_t my_pe = aclshmem_my_pe();                                                           \
            int64_t n_pes = aclshmem_n_pes();                                                           \
                                                                                                        \
            for (int64_t peer = 0; peer < n_pes; peer++) {                                              \
                if (peer == my_pe) {                                                                    \
                    continue;                                                                           \
                }                                                                                       \
                GM_ADDR src_addr = gva + peer * MESSAGE_SIZE;                                           \
                GM_ADDR dst_addr = gva + peer * MESSAGE_SIZE;                                           \
                aclshmem_##NAME##_get_nbi(                                                              \
                    reinterpret_cast<__gm__ TYPE*>(dst_addr), reinterpret_cast<__gm__ TYPE*>(src_addr), \
                    MESSAGE_SIZE / sizeof(TYPE), peer);                                                 \
                aclshmem_quiet();                                                                       \
            }                                                                                           \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    void test_rdma_##NAME##_put_nbi(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)    \
    {                                                                                                   \
        RDMA_##NAME##PutNbiTestKernel<<<block_dim, nullptr, stream>>>(gva, config);                     \
    }                                                                                                   \
                                                                                                        \
    void test_rdma_##NAME##_get_nbi(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)    \
    {                                                                                                   \
        RDMA_##NAME##GetNbiTestKernel<<<block_dim, nullptr, stream>>>(gva, config);                     \
    }

// Generate kernels for all supported types
ACLSHMEM_MEM_PUT_GET_FUNC(GEN_PUT_GET_NBI_KERNEL);
#undef GEN_PUT_GET_NBI_KERNEL

// ============================================================================
// Section: Multi-type typed PUT_NBI/GET_NBI with GlobalTensor kernels
// ============================================================================

/**
 * Macro to generate kernel and wrapper for each typed PUT_NBI/GET_NBI with GlobalTensor
 * Generates:
 * - RDMA_##NAME##PutNbiTensorTestKernel: Device kernel for typed non-blocking PUT with GlobalTensor
 * - RDMA_##NAME##GetNbiTensorTestKernel: Device kernel for typed non-blocking GET with GlobalTensor
 * - test_rdma_##NAME##_put_nbi_tensor: Host wrapper for PUT_NBI Tensor kernel
 * - test_rdma_##NAME##_get_nbi_tensor: Host wrapper for GET_NBI Tensor kernel
 */
#define GEN_PUT_GET_NBI_TENSOR_KERNEL(NAME, TYPE)                                                                  \
    extern "C" __global__ __aicore__ void RDMA_##NAME##PutNbiTensorTestKernel(GM_ADDR gva, uint64_t config)        \
    {                                                                                                              \
        util_set_ffts_config(config);                                                                              \
        AscendC::TPipe pipe;                                                                                       \
                                                                                                                   \
        if ASCEND_IS_AIV {                                                                                         \
            int64_t my_pe = aclshmem_my_pe();                                                                      \
            int64_t n_pes = aclshmem_n_pes();                                                                      \
                                                                                                                   \
            for (int64_t peer = 0; peer < n_pes; peer++) {                                                         \
                if (peer == my_pe) {                                                                               \
                    continue;                                                                                      \
                }                                                                                                  \
                GM_ADDR src_addr = gva + my_pe * MESSAGE_SIZE;                                                     \
                GM_ADDR dst_addr = gva + my_pe * MESSAGE_SIZE;                                                     \
                                                                                                                   \
                AscendC::GlobalTensor<TYPE> src_tensor;                                                            \
                src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE*>(src_addr), MESSAGE_SIZE / sizeof(TYPE)); \
                AscendC::GlobalTensor<TYPE> dst_tensor;                                                            \
                dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE*>(dst_addr), MESSAGE_SIZE / sizeof(TYPE)); \
                                                                                                                   \
                aclshmem_##NAME##_put_nbi(dst_tensor, src_tensor, MESSAGE_SIZE / sizeof(TYPE), peer);              \
                aclshmem_quiet();                                                                                  \
            }                                                                                                      \
        }                                                                                                          \
    }                                                                                                              \
                                                                                                                   \
    extern "C" __global__ __aicore__ void RDMA_##NAME##GetNbiTensorTestKernel(GM_ADDR gva, uint64_t config)        \
    {                                                                                                              \
        util_set_ffts_config(config);                                                                              \
        AscendC::TPipe pipe;                                                                                       \
                                                                                                                   \
        if ASCEND_IS_AIV {                                                                                         \
            int64_t my_pe = aclshmem_my_pe();                                                                      \
            int64_t n_pes = aclshmem_n_pes();                                                                      \
                                                                                                                   \
            for (int64_t peer = 0; peer < n_pes; peer++) {                                                         \
                if (peer == my_pe) {                                                                               \
                    continue;                                                                                      \
                }                                                                                                  \
                GM_ADDR src_addr = gva + peer * MESSAGE_SIZE;                                                      \
                GM_ADDR dst_addr = gva + peer * MESSAGE_SIZE;                                                      \
                                                                                                                   \
                AscendC::GlobalTensor<TYPE> src_tensor;                                                            \
                src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE*>(src_addr), MESSAGE_SIZE / sizeof(TYPE)); \
                AscendC::GlobalTensor<TYPE> dst_tensor;                                                            \
                dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE*>(dst_addr), MESSAGE_SIZE / sizeof(TYPE)); \
                                                                                                                   \
                aclshmem_##NAME##_get_nbi(dst_tensor, src_tensor, MESSAGE_SIZE / sizeof(TYPE), peer);              \
                aclshmem_quiet();                                                                                  \
            }                                                                                                      \
        }                                                                                                          \
    }                                                                                                              \
                                                                                                                   \
    void test_rdma_##NAME##_put_nbi_tensor(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)        \
    {                                                                                                              \
        RDMA_##NAME##PutNbiTensorTestKernel<<<block_dim, nullptr, stream>>>(gva, config);                          \
    }                                                                                                              \
                                                                                                                   \
    void test_rdma_##NAME##_get_nbi_tensor(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config)        \
    {                                                                                                              \
        RDMA_##NAME##GetNbiTensorTestKernel<<<block_dim, nullptr, stream>>>(gva, config);                          \
    }

// Generate kernels for all supported types
ACLSHMEM_MEM_PUT_GET_FUNC(GEN_PUT_GET_NBI_TENSOR_KERNEL);
#undef GEN_PUT_GET_NBI_TENSOR_KERNEL