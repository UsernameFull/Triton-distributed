/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern int32_t test_rdma_init(int pe_id, int n_pes, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

static void test_handle_wait_basic(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id = pe_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_rdma_init(pe_id, n_pes, local_mem_size, &stream);
    if (status != 0) {
        GTEST_SKIP();
        return;
    }
    ASSERT_NE(stream, nullptr);

    auto *dst_sym = reinterpret_cast<uint64_t *>(aclshmem_malloc(sizeof(uint64_t)));
    ASSERT_NE(dst_sym, nullptr);

    uint64_t *src_dev = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void **>(&src_dev), sizeof(uint64_t), ACL_MEM_MALLOC_NORMAL_ONLY), 0);

    const uint64_t kOffset = 1000;
    const uint64_t send_val = static_cast<uint64_t>(pe_id) + kOffset;
    const uint64_t zero = 0;
    ASSERT_EQ(aclrtMemcpy(dst_sym, sizeof(uint64_t), &zero, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMemcpy(src_dev, sizeof(uint64_t), &send_val, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE), 0);

    aclshmemi_control_barrier_all();

    const int dst_pe = (pe_id + 1) % n_pes;
    aclshmem_uint64_put_nbi(dst_sym, src_dev, 1, dst_pe);

    aclshmem_handle_t handle;
    handle.team_id = ACLSHMEM_TEAM_WORLD;
    aclshmemx_handle_wait(handle, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    aclshmemi_control_barrier_all();

    uint64_t got = 0;
    ASSERT_EQ(aclrtMemcpy(&got, sizeof(uint64_t), dst_sym, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
    const int src_pe = (pe_id - 1 + n_pes) % n_pes;
    const uint64_t expect = static_cast<uint64_t>(src_pe) + kOffset;
    ASSERT_EQ(got, expect);

    ASSERT_EQ(aclrtFree(src_dev), 0);
    aclshmem_free(dst_sym);

    test_finalize(stream, device_id);
}

TEST(TEST_SYNC_API, test_handle_wait_basic)
{
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_handle_wait_basic, local_mem_size, processCount);
}

