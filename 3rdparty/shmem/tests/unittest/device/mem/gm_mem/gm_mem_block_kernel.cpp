/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

static constexpr int kLength = 16;

#define KERNEL_PUT_BLOCKING(NAME, TYPE)                                                 \
    class kernel_##NAME##_put_blocking {                                                \
    public:                                                                             \
        __aicore__ inline kernel_##NAME##_put_blocking() {}                             \
        __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)                           \
        {                                                                               \
            gva_gm = (__gm__ TYPE *)gva;                                                \
            dev_gm = (__gm__ TYPE *)dev;                                                \
            pe = aclshmem_my_pe();                                                      \
            pe_size = aclshmem_n_pes();                                                 \
        }                                                                               \
        __aicore__ inline void Process(uint64_t config)                                 \
        {                                                                               \
            util_set_ffts_config(config);                                               \
            int32_t remote_pe = (int32_t)((pe + 1) % pe_size);                          \
            aclshmem_##NAME##_put(gva_gm, dev_gm, (uint32_t)kLength, remote_pe);        \
            aclshmemx_barrier_all_vec();                                                \
        }                                                                               \
    private:                                                                            \
        __gm__ TYPE *gva_gm;                                                            \
        __gm__ TYPE *dev_gm;                                                            \
        int64_t pe;                                                                     \
        int64_t pe_size;                                                                \
    }

ACLSHMEM_FUNC_TYPE_KERNEL(KERNEL_PUT_BLOCKING);

#define PUT_BLOCKING_TEST(NAME, TYPE)                                                                           \
    extern "C" __global__ __aicore__ void put_##NAME##_blocking_test(GM_ADDR gva, GM_ADDR dev, uint64_t config) \
    {                                                                                                           \
        kernel_##NAME##_put_blocking op;                                                                        \
        op.Init(gva, dev);                                                                                      \
        op.Process(config);                                                                                     \
    }

ACLSHMEM_FUNC_TYPE_KERNEL(PUT_BLOCKING_TEST);

#define TEST_PUT_BLOCKING(NAME, TYPE)                                                                               \
    void test_##NAME##_put_blocking(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, uint8_t *dev)  \
    {                                                                                                               \
        put_##NAME##_blocking_test<<<block_dim, nullptr, stream>>>(gva, dev, config);                               \
    }

ACLSHMEM_FUNC_TYPE_KERNEL(TEST_PUT_BLOCKING);

#define KERNEL_GET_BLOCKING(NAME, TYPE)                                 \
    class kernel_##NAME##_get_blocking {                                \
    public:                                                             \
        __aicore__ inline kernel_##NAME##_get_blocking() {}             \
        __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)           \
        {                                                               \
            gva_gm = (__gm__ TYPE *)gva;                                \
            dev_gm = (__gm__ TYPE *)dev;                                \
            pe_size = aclshmem_n_pes();                                 \
        }                                                               \
        __aicore__ inline void Process(uint64_t config)                 \
        {                                                               \
            util_set_ffts_config(config);                               \
            for (int32_t pe = 0; pe < (int32_t)pe_size; ++pe) {         \
                aclshmem_##NAME##_get(dev_gm + (uint32_t)kLength * pe,  \
                                      gva_gm,                           \
                                      (uint32_t)kLength,                \
                                      pe);                              \
            }                                                           \
            aclshmemx_barrier_all_vec();                                \
        }                                                               \
    private:                                                            \
        __gm__ TYPE *gva_gm;                                            \
        __gm__ TYPE *dev_gm;                                            \
        int64_t pe_size;                                                \
    }

ACLSHMEM_FUNC_TYPE_KERNEL(KERNEL_GET_BLOCKING);

#define GET_BLOCKING_TEST(NAME, TYPE)                                                                           \
    extern "C" __global__ __aicore__ void get_##NAME##_blocking_test(GM_ADDR gva, GM_ADDR dev, uint64_t config) \
    {                                                                                                           \
        kernel_##NAME##_get_blocking op;                                                                        \
        op.Init(gva, dev);                                                                                      \
        op.Process(config);                                                                                     \
    }

ACLSHMEM_FUNC_TYPE_KERNEL(GET_BLOCKING_TEST);

#define TEST_GET_BLOCKING(NAME, TYPE)                                                                               \
    void test_##NAME##_get_blocking(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, uint8_t *dev)  \
    {                                                                                                               \
        get_##NAME##_blocking_test<<<block_dim, nullptr, stream>>>(gva, dev, config);                               \
    }

ACLSHMEM_FUNC_TYPE_KERNEL(TEST_GET_BLOCKING);
