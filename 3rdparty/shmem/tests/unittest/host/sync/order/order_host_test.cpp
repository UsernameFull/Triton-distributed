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
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "order_kernel.h"

static void test_quiet_order(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    aclrtStream stream;
    int64_t device_id = rank_id % test_gnpu_num + test_first_npu;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    int total_size = 64;
    uint64_t *dev_ptr = static_cast<uint64_t *>(aclshmem_malloc(total_size * sizeof(uint64_t)));
    ASSERT_EQ(aclrtMemset(dev_ptr, total_size * sizeof(uint64_t), 0, total_size * sizeof(uint64_t)), 0);

    std::vector<uint64_t> host_buf(total_size, 0);

    std::cout << "[TEST] fence order test rank " << rank_id << std::endl;
    quiet_order_do(stream, util_get_ffts_config(), (uint8_t *)dev_ptr, rank_id, n_ranks);

    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtMemcpy(host_buf.data(), total_size * sizeof(uint64_t), dev_ptr, total_size * sizeof(uint64_t),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              0);

    if (rank_id == 1) {
        ASSERT_EQ(host_buf[33U], 0xBBu);
        ASSERT_EQ(host_buf[34U], 0xAAu);
    }

    aclshmem_free(dev_ptr);

    test_finalize(stream, device_id);
}

static void test_fence_order(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    int total_size = 64;
    uint64_t *addr_dev = static_cast<uint64_t *>(aclshmem_malloc(total_size * sizeof(uint64_t)));
    ASSERT_EQ(aclrtMemset(addr_dev, total_size * sizeof(uint64_t), 0, total_size * sizeof(uint64_t)), 0);

    std::vector<uint64_t> addr_host(total_size, 0);

    std::cout << "[TEST] fence order test rank " << rank_id << std::endl;
    fence_order_do(stream, util_get_ffts_config(), (uint8_t *)addr_dev, rank_id, n_ranks);

    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtMemcpy(addr_host.data(), total_size * sizeof(uint64_t), addr_dev, total_size * sizeof(uint64_t),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              0);

    if (rank_id == 1) {
        ASSERT_EQ(addr_host[17U], 84u);
        ASSERT_EQ(addr_host[18U], 42u);
    }
    aclshmem_free(addr_dev);

    test_finalize(stream, device_id);
}

TEST(TEST_SYNC_API, test_quiet_order)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_quiet_order, local_mem_size, process_count);
}

TEST(TEST_SYNC_API, test_fence_order)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_fence_order, local_mem_size, process_count);
}