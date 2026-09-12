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
#include "shmemi_device_common.h"

extern "C" ACLSHMEM_GLOBAL void increase(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    util_set_ffts_config(config);

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)
    // scalar unit of cube core is not affected by barrier
    aclshmem_barrier_all();
    aclshmem_barrier_all();
#endif

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    uint64_t val = aclshmemi_load((__gm__ uint64_t *)addr);

    aclshmem_barrier_all();
    GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 1) % rank_size);
    aclshmemi_store((__gm__ uint64_t *)remote, val + 1);
    aclshmem_barrier_all();
#endif
}

extern "C" ACLSHMEM_GLOBAL void increase_vec(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    util_set_ffts_config(config);

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    uint64_t val = aclshmemi_load((__gm__ uint64_t *)addr);

    aclshmemx_barrier_all_vec();
    GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 1) % rank_size);
    aclshmemi_store((__gm__ uint64_t *)remote, val + 1);
    aclshmemx_barrier_all_vec();
#endif
}

extern "C" ACLSHMEM_GLOBAL void increase_odd_team(uint64_t config, GM_ADDR addr, int rank_id,
    int rank_size, aclshmem_team_t team_id) {
    util_set_ffts_config(config);

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)
    // scalar unit of cube core is not affected by barrier
    aclshmem_barrier_all();
    aclshmem_barrier_all();
#endif

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    uint64_t val = aclshmemi_load((__gm__ uint64_t *)addr);

    aclshmem_barrier(team_id);
    if (rank_id & 1) {
        GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 2) % rank_size);
        aclshmemi_store((__gm__ uint64_t *)remote, val + 1);
    }
    aclshmem_barrier(team_id);
#endif
}

extern "C" ACLSHMEM_GLOBAL void increase_vec_odd_team(uint64_t config, GM_ADDR addr, int rank_id,
    int rank_size, aclshmem_team_t team_id) {
    util_set_ffts_config(config);

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    uint64_t val = aclshmemi_load((__gm__ uint64_t *)addr);

    aclshmemx_barrier_vec(team_id);
    if (rank_id & 1) {
        GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 2) % rank_size);
        aclshmemi_store((__gm__ uint64_t *)remote, val + 1);
    }
    aclshmemx_barrier_vec(team_id);
#endif
}

void increase_do(void* stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size)
{
    increase<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

void increase_vec_do(void* stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size)
{
    increase_vec<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

void increase_do_odd_team(void* stream, uint64_t config, uint8_t *addr, int rank_id,
    int rank_size, aclshmem_team_t team_id)
{
    increase_odd_team<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size, team_id);
}

void increase_vec_do_odd_team(void* stream, uint64_t config, uint8_t *addr, int rank_id,
    int rank_size, aclshmem_team_t team_id)
{
    increase_vec_odd_team<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size, team_id);
}