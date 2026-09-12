/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <cstdlib>
#include <string>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "team_kernel.h"

#include <gtest/gtest.h>
using namespace std;

static int32_t test_get_device_state(aclrtStream stream, uint8_t *gva, uint32_t rank_id, uint32_t rank_size,
                                     aclshmem_team_t team_id, int stride)
{
    int *y_host;
    int num3 = 3;
    int num5 = 5;
    size_t input_size = 1024 * sizeof(int);
    EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);  // size = 1024

    uint32_t block_dim = 1;
    void *ptr          = aclshmem_malloc(1024);
    int32_t device_id;
    ACLSHMEM_CHECK_RET(aclrtGetDevice(&device_id));
    get_device_state(block_dim, stream, (uint8_t *)ptr, team_id);
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(1);

    EXPECT_EQ(aclrtMemcpy(y_host, num5 * sizeof(int), ptr, num5 * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST), 0);

    if (rank_id & 1) {
        int idx = 0;
        EXPECT_EQ(y_host[idx++], rank_size);
        EXPECT_EQ(y_host[idx++], rank_id);
        EXPECT_EQ(y_host[idx++], rank_id / stride);
        EXPECT_EQ(y_host[idx++], rank_size / stride);
        EXPECT_EQ(y_host[idx++], (y_host[num3] - 1) * stride + rank_id % stride);
    }

    EXPECT_EQ(aclrtFreeHost(y_host), 0);
    return 0;
}

void test_aclshmem_team(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);
    // #################### 子通信域切分测试 ############################
    const int STRIDE_LENGTH_2 = 2;
    aclshmem_team_t team_odd;
    int start     = 1;
    int stride    = STRIDE_LENGTH_2;
    int team_size = n_ranks / stride;
    aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    aclshmem_team_t team_even;
    start     = 0;
    stride    = STRIDE_LENGTH_2;
    team_size = n_ranks / stride;
    aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_even);

    // #################### host侧取值测试 ##############################
    if (rank_id & 1) {
        ASSERT_EQ(aclshmem_team_n_pes(team_odd), team_size);
        ASSERT_EQ(aclshmem_team_my_pe(team_odd), rank_id / stride);
        ASSERT_EQ(aclshmem_n_pes(), n_ranks);
        ASSERT_EQ(aclshmem_my_pe(), rank_id);

        int local_idx  = aclshmem_team_my_pe(team_odd);
        int global_idx = aclshmem_team_translate_pe(team_odd, local_idx, ACLSHMEM_TEAM_WORLD);
        ASSERT_EQ(global_idx, rank_id);

        int back_local = aclshmem_team_translate_pe(ACLSHMEM_TEAM_WORLD, rank_id, team_odd);
        ASSERT_EQ(back_local, local_idx);

        int invalid_team = aclshmem_team_translate_pe(team_odd, local_idx, ACLSHMEM_TEAM_INVALID);
        ASSERT_EQ(invalid_team, -1);

        int invalid_src = aclshmem_team_translate_pe(team_odd, team_size, ACLSHMEM_TEAM_WORLD);
        ASSERT_EQ(invalid_src, -1);

        int invalid_dest = aclshmem_team_translate_pe(ACLSHMEM_TEAM_WORLD, start - 1, team_odd);
        ASSERT_EQ(invalid_dest, -1);

        int invalid_odd_even = aclshmem_team_translate_pe(team_odd, local_idx, team_even);
        ASSERT_EQ(invalid_odd_even, -1);
    }

    // #################### 2d子通信域切分测试 ############################
    aclshmem_team_t team_x;
    aclshmem_team_t team_y;
    int x_range   = 2;
    int y_range   = n_ranks / 2;
    int errorCode = aclshmem_team_split_2d(ACLSHMEM_TEAM_WORLD, x_range, &team_x, &team_y);
    ASSERT_EQ(errorCode, 0);

    // #################### host侧取值测试 ##############################
    ASSERT_EQ(aclshmem_team_n_pes(team_x), x_range);
    ASSERT_EQ(aclshmem_team_n_pes(team_y), y_range);
    ASSERT_EQ(aclshmem_n_pes(), n_ranks);
    ASSERT_EQ(aclshmem_my_pe(), rank_id);

    int local_x_idx  = aclshmem_team_my_pe(team_x);
    int global_x_idx = aclshmem_team_translate_pe(team_x, local_x_idx, ACLSHMEM_TEAM_WORLD);
    ASSERT_EQ(global_x_idx, rank_id);

    int back_x_local = aclshmem_team_translate_pe(ACLSHMEM_TEAM_WORLD, rank_id, team_x);
    ASSERT_EQ(back_x_local, local_x_idx);

    int local_y_idx  = aclshmem_team_my_pe(team_y);
    int global_y_idx = aclshmem_team_translate_pe(team_y, local_y_idx, ACLSHMEM_TEAM_WORLD);
    ASSERT_EQ(global_y_idx, rank_id);

    int back_y_local = aclshmem_team_translate_pe(ACLSHMEM_TEAM_WORLD, rank_id, team_y);
    ASSERT_EQ(back_y_local, local_y_idx);

    int x_to_y_id = aclshmem_team_translate_pe(team_x, local_x_idx, team_y);
    ASSERT_NE(x_to_y_id, -1);

    // #################### device代码测试 ##############################

    auto status = test_get_device_state(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks, team_odd, stride);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);

    // #################### 相关资源释放 ################################
    aclshmem_team_destroy(team_odd);

    std::cerr << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
}

TEST(TestTeamApi, TestShmemTeam)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_team, local_mem_size, process_count);
}

void test_aclshmem_team_config(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    aclshmem_team_t team_odd;
    int start     = 0;
    int stride    = 1;
    int team_size = n_ranks;
    int ret       = aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(team_odd >= 0 && team_odd < ACLSHMEM_MAX_TEAMS);

    aclshmem_team_config_t config;
    config.num_contexts = 1;
    ret                 = aclshmem_team_get_config(team_odd, &config);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(config.num_contexts, 0);

    aclshmem_team_destroy(team_odd);

    std::cerr << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
}

TEST(TestTeamApi, TestShmemTeamConfig)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_team_config, local_mem_size, process_count);
}

TEST(TestTeamApi, TestShmemTeamSplitStrided_failConditions)
{
    int ret = aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, 0, 1, 1, nullptr);
    EXPECT_EQ(ret, ACLSHMEM_INVALID_PARAM);

    aclshmem_team_t team_odd;
    ret = aclshmem_team_split_strided(-1, 0, 1, 1, &team_odd);
    EXPECT_EQ(ret, ACLSHMEM_INVALID_PARAM);

    const int32_t pe_size = 2;
    ret = aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, 0, -1, pe_size, &team_odd);
    EXPECT_EQ(ret, ACLSHMEM_INVALID_PARAM);

    const int32_t stride = 2;
    const int32_t pe_start = ACLSHMEM_MAX_PES - 1;
    ret = aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, pe_start, stride, pe_size, &team_odd);
    EXPECT_EQ(ret, ACLSHMEM_INVALID_PARAM);
}

TEST(TestTeamApi, ShmemTeamSplit2d_failConditions)
{
    aclshmem_team_t team_x;
    aclshmem_team_t team_y;
    int x_range   = 2;
    int errorCode = aclshmem_team_split_2d(ACLSHMEM_TEAM_WORLD, x_range, &team_x, nullptr);
    EXPECT_EQ(errorCode, ACLSHMEM_INVALID_PARAM);

    errorCode = aclshmem_team_split_2d(ACLSHMEM_TEAM_WORLD, 0, &team_x, nullptr);
    EXPECT_EQ(errorCode, ACLSHMEM_INVALID_PARAM);

    errorCode = aclshmem_team_split_2d(-1, 0, &team_x, nullptr);
    EXPECT_EQ(errorCode, ACLSHMEM_INVALID_PARAM);
}