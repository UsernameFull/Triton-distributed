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
#include "shmemi_host_common.h"
#include "opdev/bfloat16.h"
#include "opdev/fp16_t.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

const int input_length = 16;
const int test_offset = 10;

extern void test_shmemx_mte_put(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, uint8_t *dev_ptr);
extern void test_shmemx_mte_get(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, uint8_t *dev_ptr);

static void test_shmemx_mte_put_get(aclrtStream stream, uint8_t *gva, uint32_t rank_id, uint32_t rank_size)
{
    int total_size = input_length * static_cast<int>(rank_size);
    size_t input_size = total_size * sizeof(int8_t);

    std::vector<int8_t> input(total_size, 0);
    for (int i = 0; i < input_length; i++) {
        input[i] = static_cast<int8_t>(rank_id + test_offset);
    }

    void *dev_ptr;
    ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);

    ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    uint32_t block_dim = 1;
    void *ptr = aclshmem_malloc(input_size);
    test_shmemx_mte_put(block_dim, stream, util_get_ffts_config(), (uint8_t *)ptr, (uint8_t *)dev_ptr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    test_shmemx_mte_get(block_dim, stream, util_get_ffts_config(), (uint8_t *)ptr, (uint8_t *)dev_ptr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, dev_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    /* result check */
    int32_t flag = 0;
    for (int i = 0; i < total_size; i++) {
        int stage = i / input_length;
        if (input[i] != static_cast<int8_t>(stage + test_offset)) {
            flag = 1;
        }
    }
    ASSERT_EQ(flag, 0);
}

void test_shmemx_mte_mem(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    test_shmemx_mte_put_get(stream, (uint8_t *)g_state.heap_base, rank_id, n_ranks);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemGmDisableL2)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmemx_mte_mem, local_mem_size, process_count);
}