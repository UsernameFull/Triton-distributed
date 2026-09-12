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
#include "gm2gm/shmemi_device_cc.h"

extern "C" ACLSHMEM_GLOBAL void quiet_order(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    util_set_ffts_config(config);
    __gm__ uint64_t *base = reinterpret_cast<__gm__ uint64_t*>(addr);
    if (rank_id == 0) {
        aclshmemi_store<uint64_t>(base, 0xAA);
        aclshmem_quiet();
        aclshmemi_store<uint64_t>(base + 32, 0xBB);
        dcci_cacheline((__gm__ uint8_t *)(base + 32));
    }

    if (rank_id == 1) {
        uint64_t seen_b;
        __gm__ uint64_t *remote = (__gm__ uint64_t *)aclshmem_ptr(base, 0);
        do {
            dcci_cacheline((__gm__ uint8_t *)(remote + 32));
            seen_b = aclshmemi_load<uint64_t>(remote + 32);
        } while (seen_b != 0xBB);
        dcci_cacheline((__gm__ uint8_t *)(remote));
        uint64_t seen_a = aclshmemi_load<uint64_t>(remote);

        aclshmemi_store<uint64_t>(base + 33, seen_b);
        aclshmemi_store<uint64_t>(base + 34, seen_a);
    }
}

extern "C" ACLSHMEM_GLOBAL void fence_order(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    util_set_ffts_config(config);
    __gm__ uint64_t *base = reinterpret_cast<__gm__ uint64_t*>(addr);
    if (rank_id == 0) {
        uint64_t a_val = 42, b_val = 84;
        aclshmemi_store<uint64_t>(base, a_val);
        aclshmem_fence();
        aclshmemi_store<uint64_t>(base + 16, b_val);
        dcci_cacheline((__gm__ uint8_t *)(base + 16));
    }

    if (rank_id == 1) {
        uint64_t seen_b;
        __gm__ uint64_t *remote = (__gm__ uint64_t *)aclshmem_ptr(base, 0);
        do {
            dcci_cacheline((__gm__ uint8_t *)(remote + 16));
            seen_b = aclshmemi_load<uint64_t>(remote + 16);
        } while (seen_b != 84);
        dcci_cacheline((__gm__ uint8_t *)remote);
        uint64_t seen_a = aclshmemi_load<uint64_t>(remote);

        aclshmemi_store<uint64_t>(base + 17, seen_b);
        aclshmemi_store<uint64_t>(base + 18, seen_a);
    }
}

void quiet_order_do(void* stream, uint64_t config, uint8_t *addr, int32_t rank_id, int32_t n_ranks)
{
    quiet_order<<<1, nullptr, stream>>>(config, addr, rank_id, n_ranks);
}

void fence_order_do(void* stream, uint64_t config, uint8_t *addr, int32_t rank_id, int32_t n_ranks)
{
    fence_order<<<1, nullptr, stream>>>(config, addr, rank_id, n_ranks);
}