/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SYNC_UTIL_H
#define SYNC_UTIL_H
#include "kernel_operator.h"
using namespace AscendC;

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
constexpr int32_t BUFF_SIZE = 500 * 1024 * 1024;
constexpr int32_t FLAG_OFFSET = 500 * 1024 * 1024 / sizeof(int32_t);


template<typename T>
FORCE_INLINE_AICORE void gm_store(__gm__ T *addr, T val) {
    *((__gm__ T *)addr) = val;
}

template<typename T>
FORCE_INLINE_AICORE T gm_load(__gm__ T *cache) {
    return *((__gm__ T *)cache);
}

FORCE_INLINE_AICORE void gm_dcci(__gm__ uint8_t * addr) {
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

FORCE_INLINE_AICORE int32_t gm_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    do {
        gm_dcci((__gm__ uint8_t *)sig_addr);

        if (*sig_addr == cmp_val) {
            return *sig_addr;
        }

        // in case when peer pe enters next barrier
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true);

    return -1;
}
 
FORCE_INLINE_AICORE void CrossRankSync(GM_ADDR symmetricPtr, int32_t rank, int32_t rankSize)
{
    // 全核同步
    __gm__ int32_t* sync_counter = (__gm__ int32_t*)symmetricPtr + FLAG_OFFSET;
    __gm__ int32_t* sync_base = (__gm__ int32_t*)symmetricPtr + FLAG_OFFSET + 1024;
    int count = gm_load(sync_base) + 1;
    int vec_id = AscendC::GetBlockIdx();
    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    for (int i = vec_id; i < rankSize; i += vec_size) {
        __gm__ int32_t* sync_remote = (__gm__ int32_t*)(aclshmem_ptr(symmetricPtr, i)) + FLAG_OFFSET + rank * 16;
        gm_store(sync_remote, count);
        gm_dcci((__gm__ uint8_t*)sync_remote);
        auto sync_check = sync_counter + i * 16;
        gm_signal_wait_until_eq_for_barrier(sync_check, count);
    }

    AscendC::SyncAll<true>();
    gm_store(sync_base, count);
}

// 全卡同步接口
FORCE_INLINE_AICORE void CrossRankSyncV1(GM_ADDR symmetricPtr, int32_t rank, int32_t rankSize)
{
    // 全核同步
    AscendC::SyncAll<true>();
    __gm__ int32_t* sync_counter = (__gm__ int32_t*)symmetricPtr + FLAG_OFFSET;
    __gm__ int32_t* sync_base = (__gm__ int32_t*)symmetricPtr + FLAG_OFFSET + 32;
    int count = gm_load(sync_base) + 1;
    int vec_id = AscendC::GetBlockIdx();
    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    for (int i = vec_id; i < rankSize; i += vec_size) {
        __gm__ int32_t* sync_remote = (__gm__ int32_t*)(aclshmem_ptr(symmetricPtr, i)) + FLAG_OFFSET + rank * 16;
        gm_store(sync_remote, count);
        gm_dcci((__gm__ uint8_t*)sync_remote);
        auto sync_check = sync_counter + i * 16;
        gm_signal_wait_until_eq_for_barrier(sync_check, count);
    }

    AscendC::SyncAll<true>();
    gm_store(sync_base, count);
}
#endif