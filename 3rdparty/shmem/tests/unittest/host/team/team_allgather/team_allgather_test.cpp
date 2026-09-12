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
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "team_allgather_kernel.h"

void test_aclshmem_team_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    aclshmem_team_t team_odd;
    int start     = 1;
    int stride    = 2;
    int team_size = n_ranks / 2;
    void *ptr     = aclshmem_malloc(1024);
    if (rank_id & 1) {
        // Team split
        aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

        // Initialize data
        uint32_t trans_size = 16;
        int num11 = 11;
        int num10 = 10;
        std::vector<int32_t> input(trans_size, 0);
        for (int i = 0; i < trans_size; i++) {
            input[i] = (rank_id + num10);
        }

        ASSERT_EQ(aclrtMemcpy((void *)((uint64_t)ptr + aclshmem_team_my_pe(team_odd) * trans_size * sizeof(int32_t)),
                              trans_size, input.data(), trans_size, ACL_MEMCPY_HOST_TO_DEVICE),
                  0);

        // Execute AllGather
        team_allgather(1, stream, util_get_ffts_config(), (uint8_t *)ptr, team_odd);
        EXPECT_EQ(aclrtSynchronizeStream(stream), 0);

        // Check results
        int32_t *y_host;
        size_t input_size = team_size * trans_size * sizeof(int32_t);
        EXPECT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size), 0);
        EXPECT_EQ(aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

        int multiple = 2;
        for (int i = 0; i < team_size; i++) {
            EXPECT_EQ(y_host[trans_size * i], num11 + i * multiple);
        }

        EXPECT_EQ(aclrtFreeHost(y_host), 0);
        aclshmem_team_destroy(team_odd);
    }
    aclshmem_free(ptr);

    std::cerr << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
}

TEST(TestTeamFunc, TestShmemTeam)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_team_all_gather, local_mem_size, process_count);
}