/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "gm2gm/shmemi_device_cc_kernel.h"
using namespace std;

uint64_t g_team_mask = 0;
aclshmemx_team_t *g_aclshmem_team_pool = nullptr;

#define ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(func_name)                                                              \
    do {                                                                                                            \
        if (g_instance_ctx->id != 0) {                                                                              \
            SHM_LOG_ERROR((func_name) << " is not supported in multi-instance mode (instance_id="                   \
                            << g_instance_ctx->id << "). "                                                          \
                            << "Only Instance 0 supports team operations. Please Create new instance instead.");    \
            return ACLSHMEM_NOT_SUPPORTED;                                                                          \
        }                                                                                                           \
    } while (0)                                                                                                     

inline std::string team_config2string(aclshmemx_team_t *config)
{
    std::ostringstream oss;
    oss << "[team:" << config->team_idx;
    oss << ",npes:" << config->size;
    oss << ",mype:" << config->mype;
    oss << ",start:" << config->start;
    oss << ",stride:" << config->stride;
    oss << "]";
    return oss.str();
}

inline bool is_valid_team(aclshmem_team_t &team)
{
    return (g_state.is_aclshmem_initialized && g_aclshmem_team_pool != nullptr && team >= 0 && team < ACLSHMEM_MAX_TEAMS &&
            ((g_team_mask >> team) & 1));
}

inline void device_team_destroy(int32_t team_idx)
{
    // device_ptr Free
    aclshmemx_team_t *device_team_ptr = g_state.team_pools[team_idx];
    if (device_team_ptr != nullptr) {
        if (aclrtFree((void *)device_team_ptr) != ACL_SUCCESS) {
            SHM_LOG_ERROR("aclrtFree for device_team_ptr failed for team " << team_idx);
        }
        g_state.team_pools[team_idx] = nullptr;
    }
}

inline int32_t device_team_update(int team_idx, aclshmemx_team_t *host_team_ptr)
{
    // device_ptr Malloc
    void *team_ptr = nullptr;
    ACLSHMEM_CHECK_RET(aclrtMalloc(&team_ptr, sizeof(aclshmemx_team_t), ACL_MEM_MALLOC_NORMAL_ONLY));
    auto ret = aclrtMemcpy((aclshmemx_team_t *)team_ptr, sizeof(aclshmemx_team_t), host_team_ptr, sizeof(aclshmemx_team_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR("memcpy device team info failed, ret: " << ret);
        ACLSHMEM_CHECK_RET(aclrtFree(team_ptr));
        return ACLSHMEM_INNER_ERROR;
    }
    g_state.team_pools[team_idx] = (aclshmemx_team_t *)team_ptr;
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_team_init_sync_pool()
{
    g_state.sync_pool = (uint64_t)aclshmem_malloc(SYNC_POOL_SIZE);
    if (g_state.sync_pool == 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("malloc sync pool failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    auto ret = aclrtMemset((void *)g_state.sync_pool, SYNC_POOL_SIZE, 0, SYNC_POOL_SIZE);
    if (ret != 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("memset sync pool failed. ret=" << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_team_init_sync_counter()
{
    g_state.sync_counter = (uint64_t)aclshmem_malloc(SYNC_COUNTERS_SIZE);
    if (g_state.sync_counter == 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("malloc sync counter failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    auto ret = aclrtMemset((void *)g_state.sync_counter, SYNC_COUNTERS_SIZE, 0, SYNC_COUNTERS_SIZE);
    if (ret != 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("memset sync counter failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_team_init_core_sync_pool()
{
    auto ret = aclrtMalloc((void **)&(g_state.core_sync_pool), ACLSHMEM_CORE_SYNC_POOL_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || g_state.core_sync_pool == 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("malloc core sync pool failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    ret = aclrtMemset((void *)g_state.core_sync_pool, ACLSHMEM_CORE_SYNC_POOL_SIZE, 0, ACLSHMEM_CORE_SYNC_POOL_SIZE);
    if (ret != 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("memset core sync pool failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_team_init_core_sync_counter()
{
    auto ret = aclrtMalloc((void **)&(g_state.core_sync_counter), ACLSHMEM_CORE_SYNC_COUNTER_SIZE,
        ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || g_state.core_sync_counter == 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("malloc core sync counter failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    ret = aclrtMemset((void *)g_state.core_sync_counter, ACLSHMEM_CORE_SYNC_COUNTER_SIZE, 0, ACLSHMEM_CORE_SYNC_COUNTER_SIZE);
    if (ret != 0) {
        aclshmemi_team_finalize();
        SHM_LOG_ERROR("memset core sync counter failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_team_init(int32_t rank, int32_t size)
{
    /* Initialize ACLSHMEM_TEAM_WORLD */
    int team_size = g_instance_ctx->id == 0 ? ACLSHMEM_MAX_TEAMS : 1;
    g_aclshmem_team_pool = (aclshmemx_team_t *)calloc(team_size, sizeof(aclshmemx_team_t));
    if (g_aclshmem_team_pool == nullptr) {
        SHM_LOG_ERROR("malloc host shmem team pool failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    for (int i = 0; i < team_size; i++) {
        g_aclshmem_team_pool[i] = aclshmemx_team_t{-1, -1, -1, -1, -1, {0, 0, 0,{'0'}}, {-1}};
    }

    aclshmemx_team_t &aclshmem_team_world = g_aclshmem_team_pool[ACLSHMEM_TEAM_WORLD];
    aclshmem_team_world.team_idx = ACLSHMEM_TEAM_WORLD;
    aclshmem_team_world.start = 0;
    aclshmem_team_world.stride = 1;
    aclshmem_team_world.size = size;
    aclshmem_team_world.mype = rank;
    g_team_mask |= 1ULL << ACLSHMEM_TEAM_WORLD;

    aclshmemi_team_populate_pe_mappings_from_constant_stride(&aclshmem_team_world);
    ACLSHMEM_CHECK_RET(device_team_update(ACLSHMEM_TEAM_WORLD, &aclshmem_team_world));

    /* Initialize TEAM SYNC */
    auto ret = aclshmemi_team_init_sync_pool();
    if (ret != 0) {
        return ret;
    }

    ret = aclshmemi_team_init_sync_counter();
    if (ret != 0) {
        return ret;
    }

    ret = aclshmemi_team_init_core_sync_pool();
    if (ret != 0) {
        return ret;
    }

    return aclshmemi_team_init_core_sync_counter();
}

int32_t first_free_idx_fetch()
{
    int32_t aclshmem_max_teams = ACLSHMEM_MAX_TEAMS;
    for (int32_t i = 0; i < aclshmem_max_teams; i++) {
        if (!((g_team_mask >> i) & 1)) {
            g_team_mask |= 1ULL << i;
            return i;
        }
    }
    return -1;
}

int32_t aclshmemi_team_finalize()
{
    /* Destroy all undestroyed teams */
    int team_size = g_instance_ctx->id == 0 ? ACLSHMEM_MAX_TEAMS : 1;
    for (int32_t i = 0; i < team_size; i++) {
        aclshmem_team_t team = i;
        if (is_valid_team(team)) {
            aclshmem_team_destroy(team);
        }
    }

    if (g_state.sync_counter != 0) {
        aclshmem_free(reinterpret_cast<void *>(g_state.sync_counter));
        g_state.sync_counter = 0;
    }
    if (g_state.sync_pool != 0) {
        aclshmem_free(reinterpret_cast<void *>(g_state.sync_pool));
        g_state.sync_pool = 0;
    }
    if (g_state.core_sync_counter != 0) {
        ACLSHMEM_CHECK_RET(aclrtFree(reinterpret_cast<void *>(g_state.core_sync_counter)));
        g_state.core_sync_counter = 0;
    }
    if (g_state.core_sync_pool != 0) {
        ACLSHMEM_CHECK_RET(aclrtFree(reinterpret_cast<void *>(g_state.core_sync_pool)));
        g_state.core_sync_pool = 0;
    }
    if (g_aclshmem_team_pool != nullptr) {
        free(g_aclshmem_team_pool);
        g_aclshmem_team_pool = nullptr;
    }
    return 0;
}
void aclshmemi_team_populate_from_world_pe_mapping(aclshmemx_team_t *team)
{
    for (int i = 0; i < team->size; i++) {
        int global_pe_index = team->pe_mapping[i];
        team->pe_mapping[global_pe_index + ACLSHMEM_MAX_PES] = i;
    }
}

void aclshmemi_team_populate_pe_mappings_from_constant_stride(aclshmemx_team_t *team)
{
    for (int i = 0; i < team->size; i++) {
        int global_pe_index = team->start + i * team->stride;
        team->pe_mapping[i] = global_pe_index;
    }
    aclshmemi_team_populate_from_world_pe_mapping(team);
}

int32_t aclshmemi_team_pe_mapping(aclshmem_team_t team, int pe)
{
    if (!is_valid_team(team)) {
        SHM_LOG_ERROR("input team is invalid!, team: " << team);
        return ACLSHMEM_INVALID_PARAM;
    }
    return g_aclshmem_team_pool[team].pe_mapping[pe];
}

int32_t aclshmem_team_split_strided_precheck(aclshmem_team_t parent_team, int32_t pe_start, int32_t pe_stride,
                                          int32_t pe_size, aclshmem_team_t *new_team)
{
    if (new_team == nullptr) {
        SHM_LOG_ERROR("output team is null.");
        return ACLSHMEM_INVALID_PARAM;
    }

    *new_team = ACLSHMEM_TEAM_INVALID;
    if (!is_valid_team(parent_team)) {
        SHM_LOG_ERROR("input parent team is invalid!, team: " << parent_team);
        return ACLSHMEM_INVALID_PARAM;
    }

    aclshmemx_team_t *src_team = &g_aclshmem_team_pool[parent_team];

    if (pe_start >= ACLSHMEM_MAX_PES || pe_stride >= ACLSHMEM_MAX_PES || pe_size > ACLSHMEM_MAX_PES) {
        SHM_LOG_ERROR("create team failed, input invalid, pe_start:" << pe_start << " pe_size:" << pe_size
                                                                     << " pe_stride:" << pe_stride << " parent:"
                                                                     << team_config2string(src_team));
        return ACLSHMEM_INVALID_PARAM;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmem_team_split_strided(aclshmem_team_t parent_team, int32_t pe_start, int32_t pe_stride, int32_t pe_size,
                                 aclshmem_team_t *new_team)
{
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    auto ret = aclshmem_team_split_strided_precheck(parent_team, pe_start, pe_stride, pe_size, new_team);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    aclshmemx_team_t *src_team = &g_aclshmem_team_pool[parent_team];
    if (pe_start < 0 || pe_start >= src_team->size || pe_size <= 0 || pe_size > src_team->size || pe_stride < 1) {
        SHM_LOG_ERROR("create team failed, input invalid, pe_start:" << pe_start << " pe_size:" << pe_size
                                                                     << " pe_stride:" << pe_stride << " parent:"
                                                                     << team_config2string(src_team));
        return ACLSHMEM_INVALID_PARAM;
    }

    int32_t global_pe = src_team->mype;
    int32_t global_pe_start = src_team->pe_mapping[pe_start];
    int32_t global_pe_stride = src_team->stride * pe_stride;
    int32_t global_pe_end = global_pe_start + global_pe_stride * (pe_size - 1);

    if (global_pe_start >= aclshmem_n_pes() || global_pe_end >= aclshmem_n_pes()) {
        SHM_LOG_ERROR("create team failed, large than world size, pe_start:"
                      << pe_start << " pe_size:" << pe_size << " pe_stride:" << pe_stride
                      << " world_size:" << aclshmem_n_pes() << " parent:" << team_config2string(src_team));
        return ACLSHMEM_INVALID_PARAM;
    }

    aclshmemx_team_t my_team;
    my_team.mype = (global_pe - global_pe_start) / global_pe_stride;

    if (global_pe < global_pe_start || (global_pe - global_pe_start) % global_pe_stride || my_team.mype >= pe_size) {
        SHM_LOG_INFO("This PE is not a member of the new team.");
        return ACLSHMEM_SUCCESS;
    }

    my_team.start = global_pe_start;
    my_team.stride = global_pe_stride;
    my_team.size = pe_size;

    for (int i = 0; i < pe_size; i++) {
        int temp_global_pe = global_pe_start + i * global_pe_stride;
        my_team.pe_mapping[i] = temp_global_pe;
        my_team.pe_mapping[temp_global_pe + ACLSHMEM_MAX_PES] = i;
    }

    my_team.team_idx = first_free_idx_fetch();
    if (my_team.team_idx == -1) {
        SHM_LOG_ERROR("create team failed, team num is full!");
        return ACLSHMEM_INNER_ERROR;
    }

    g_aclshmem_team_pool[my_team.team_idx] = my_team;
    if (device_team_update(my_team.team_idx, &g_aclshmem_team_pool[my_team.team_idx]) != 0) {
        aclshmem_team_destroy(my_team.team_idx);
        SHM_LOG_ERROR("create team failed, malloc device state failed!");
        return ACLSHMEM_INNER_ERROR;
    }
    if (update_device_state() != 0) {
        aclshmem_team_destroy(my_team.team_idx);
        SHM_LOG_ERROR("create team failed, update state failed!");
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_INFO("create team success:" << team_config2string(&my_team));
    *new_team = my_team.team_idx;
    return 0;
}

int aclshmemi_team_split_2d_precheck(aclshmem_team_t p_team, int x_range, aclshmem_team_t *x_team, aclshmem_team_t *y_team)
{
    if (x_team == nullptr || y_team == nullptr) {
        SHM_LOG_ERROR("output team is null.");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (x_range <= 0) {
        SHM_LOG_ERROR("input x range must be larger than 0.");
        return ACLSHMEM_INVALID_PARAM;
    }

    *x_team = ACLSHMEM_TEAM_INVALID;
    *y_team = ACLSHMEM_TEAM_INVALID;
    if (!is_valid_team(p_team)) {
        SHM_LOG_ERROR("input parent team is invalid!, team: " << p_team);
        return ACLSHMEM_INVALID_PARAM;
    }

    return ACLSHMEM_SUCCESS;
}

int aclshmemi_team_split_2d_x(aclshmem_team_t &parent_team, int32_t &x_team_counts, int32_t &src_size,
                           int &x_range, aclshmem_team_t *&x_team)
{
    int start = 0;
    aclshmemx_team_t *src_team = &g_aclshmem_team_pool[parent_team];

    for (int i = 0; i < x_team_counts; ++i) {
        aclshmem_team_t my_xteam;
        int x_stride = 1;
        int x_size = ((i == x_team_counts - 1) && (src_size % x_range != 0)) ? src_size % x_range : x_range;
        if (aclshmem_team_split_strided(parent_team, start, x_stride, x_size, &my_xteam) != ACLSHMEM_SUCCESS) {
            SHM_LOG_INFO("create x-axis team " << (i + 1) << " of " << x_team_counts << " failed");
        }
        start += x_range;

        if (my_xteam != ACLSHMEM_TEAM_INVALID) {
            if (*x_team == ACLSHMEM_TEAM_INVALID) {
                *x_team = my_xteam;
                SHM_LOG_INFO("Current pe is " << src_team->mype << " , split x-axis succeed for x- " << i);
            } else {
                return ACLSHMEM_INNER_ERROR;
            }
        }
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_team_split_2d_y(aclshmem_team_t &parent_team, int32_t &y_team_counts, int32_t &src_size,
                           int &x_range, aclshmem_team_t *&y_team)
{
    int start = 0;
    int errorCode = 0;
    aclshmemx_team_t *src_team = &g_aclshmem_team_pool[parent_team];

    for (int i = 0; i < y_team_counts; ++i) {
        aclshmem_team_t my_yteam;
        int y_stride = x_range;
        int remainder = src_size % x_range;
        int y_range = src_size / x_range;
        int y_size = (remainder && i < remainder) ? y_range + 1 : y_range;

        if (aclshmem_team_split_strided(parent_team, start, y_stride, y_size, &my_yteam) != ACLSHMEM_SUCCESS) {
            SHM_LOG_INFO("create y-axis team " << (i + 1) << " of " << y_team_counts << " failed");
        }

        start += 1;
        if (my_yteam != ACLSHMEM_TEAM_INVALID) {
            if (*y_team == ACLSHMEM_TEAM_INVALID) {
                *y_team = my_yteam;
                SHM_LOG_INFO("Current pe is " << src_team->mype << " , split y-axis succeed for y- " << i);
            } else {
                return ACLSHMEM_INNER_ERROR;
            }
        }
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmem_team_split_2d(aclshmem_team_t parent_team, int x_range, aclshmem_team_t *x_team, aclshmem_team_t *y_team)
{
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    auto ret = aclshmemi_team_split_2d_precheck(parent_team, x_range, x_team, y_team);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    aclshmemx_team_t *src_team = &g_aclshmem_team_pool[parent_team];
    int32_t src_start = src_team->start;
    int32_t src_stride = src_team->stride;
    int32_t src_size = src_team->size;
    if (x_range > src_size) {
        x_range = src_size;
    }
    int32_t x_team_counts = static_cast<int32_t>(std::ceil(src_size / static_cast<double>(x_range)));
    int32_t y_team_counts = x_range;

    ret = aclshmemi_team_split_2d_x(parent_team, x_team_counts, src_size, x_range, x_team);
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }

    return aclshmemi_team_split_2d_y(parent_team, y_team_counts, src_size, x_range, y_team);
}

int32_t aclshmem_team_translate_pe(aclshmem_team_t src_team, int32_t src_pe, aclshmem_team_t dest_team)
{
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    if (!is_valid_team(src_team) || !is_valid_team(dest_team)) {
        return -1;
    }

    if (src_pe < 0 || src_pe >= g_aclshmem_team_pool[src_team].size) {
        return -1;
    }
    int global_pe = g_aclshmem_team_pool[src_team].pe_mapping[src_pe];
    return g_aclshmem_team_pool[dest_team].pe_mapping[global_pe + ACLSHMEM_MAX_PES];
}

void aclshmem_team_destroy(aclshmem_team_t team)
{
    if (!is_valid_team(team)) {
        SHM_LOG_INFO("input team is invalid!, team: " << team);
        return;
    }

    device_team_destroy(team);
    g_team_mask ^= 1ULL << team;
    if (update_device_state() != ACLSHMEM_SUCCESS) {
        SHM_LOG_INFO("update state failed when destroy team!");
    }
}

int32_t aclshmem_my_pe(void)
{
    return g_state.mype;
}

int32_t aclshmem_n_pes(void)
{
    return g_state.npes;
}

int32_t aclshmem_team_my_pe(aclshmem_team_t team)
{
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    if (is_valid_team(team)) {
        return g_aclshmem_team_pool[team].mype;
    } else {
        return -1;
    }
}

int32_t aclshmem_team_n_pes(aclshmem_team_t team)
{
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    if (is_valid_team(team)) {
        return g_aclshmem_team_pool[team].size;
    } else {
        return -1;
    }
}

int aclshmem_team_get_config(aclshmem_team_t team, aclshmem_team_config_t *config)
{
    // Not Single Instance Mode
    ACLSHMEM_TEAM_CHECK_SINGLE_INSTANCE(__func__);

    ACLSHMEM_CHECK_RET(config == nullptr);
    if (is_valid_team(team)) {
        *config = g_aclshmem_team_pool[team].config;
        return 0;
    } else {
        return ACLSHMEM_INVALID_PARAM;
    }
}
