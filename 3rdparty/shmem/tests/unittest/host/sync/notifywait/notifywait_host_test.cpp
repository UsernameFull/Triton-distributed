/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <cstring>

#include "acl/acl.h"
#include "shmemi_host_common.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);
extern int32_t test_sdma_init(int my_pe, int n_pes, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

extern void test_put_notify_wait(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config);
extern void test_get_notify_wait(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config);
extern void test_put_tensor_notify_wait(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config);
extern void test_get_tensor_notify_wait(uint32_t block_dim, void *stream, uint8_t *gva, uint64_t config);
extern void copy_demo(uint32_t block_dim, void* stream, uint8_t* src, uint8_t* dst, int elements);

static void test_put_get_notify_wait(aclrtStream stream, uint8_t *gva, uint8_t *copy_ptr, uint32_t my_pe, uint32_t pe_size)
{
    size_t message_size = 64;
    uint32_t pe_offset = 10;
    uint32_t *in_host;
    uint32_t *out_host;
    size_t total_size = message_size * pe_size;
    uint32_t block_dim = 1;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&in_host), total_size), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&out_host), total_size), 0);
    bzero(in_host, total_size);
    for (uint32_t i = 0; i < message_size / sizeof(uint32_t); i++) {
        in_host[i + my_pe * message_size / sizeof(uint32_t)] = my_pe + pe_offset;
    }

    // Test SDMA Put (pointer interface) with notify wait
    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host, total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_put_notify_wait(block_dim, stream, (uint8_t *)gva, util_get_ffts_config());
    for (uint32_t i = 0; i < block_dim * 2; i++) {
        aclrtWaitAndResetNotify(g_state_host.notify_arr[i], g_state_host.default_stream, 0);
    }
    aclshmem_barrier_all();
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    copy_demo(1, g_state_host.default_stream, (uint8_t *)gva, copy_ptr, total_size);
    ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);

    ASSERT_EQ(aclrtMemcpy(out_host, total_size, copy_ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        ASSERT_EQ(out_host[i * message_size / sizeof(uint32_t)], i + pe_offset);
    }

    // Test SDMA Get (pointer interface) with notify wait
    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host, total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_get_notify_wait(block_dim, stream, (uint8_t *)gva, util_get_ffts_config());
    for (uint32_t i = 0; i < block_dim * 2; i++) {
        aclrtWaitAndResetNotify(g_state_host.notify_arr[i], g_state_host.default_stream, 0);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    copy_demo(1, g_state_host.default_stream, (uint8_t *)gva, copy_ptr, total_size);
    ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);

    ASSERT_EQ(aclrtMemcpy(out_host, total_size, copy_ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        ASSERT_EQ(out_host[i * message_size / sizeof(uint32_t)], i + pe_offset);
    }

    // Test SDMA Put with Tensor interface with notify wait
    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host, total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_put_tensor_notify_wait(block_dim, stream, (uint8_t *)gva, util_get_ffts_config());
    for (uint32_t i = 0; i < block_dim * 2; i++) {
        aclrtWaitAndResetNotify(g_state_host.notify_arr[i], g_state_host.default_stream, 0);
    }
    aclshmem_barrier_all();
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    copy_demo(1, g_state_host.default_stream, (uint8_t *)gva, copy_ptr, total_size);
    ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);

    ASSERT_EQ(aclrtMemcpy(out_host, total_size, copy_ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);   
    for (uint32_t i = 0; i < pe_size; i++) {
        ASSERT_EQ(out_host[i * message_size / sizeof(uint32_t)], i + pe_offset);
    }

    // Test SDMA Get with Tensor interface with notify wait
    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host, total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_get_tensor_notify_wait(block_dim, stream, (uint8_t *)gva, util_get_ffts_config());
    for (uint32_t i = 0; i < block_dim * 2; i++) {
        aclrtWaitAndResetNotify(g_state_host.notify_arr[i], g_state_host.default_stream, 0);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    copy_demo(1, g_state_host.default_stream, (uint8_t *)gva, copy_ptr, total_size);
    ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);

    ASSERT_EQ(aclrtMemcpy(out_host, total_size, copy_ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);   
    for (uint32_t i = 0; i < pe_size; i++) {
        ASSERT_EQ(out_host[i * message_size / sizeof(uint32_t)], i + pe_offset);
    }

    ASSERT_EQ(aclrtFreeHost(in_host), 0);
    ASSERT_EQ(aclrtFreeHost(out_host), 0);
}

void test_aclshmem_notify_wait(int my_pe, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id = my_pe % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_sdma_init(my_pe, n_pes, local_mem_size, &stream);
    if (status != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    size_t message_size = 64;
    size_t total_size = message_size * n_pes;
    void *ptr = aclshmem_malloc(total_size * 2);
    ASSERT_NE(ptr, nullptr);
    // Set copy_ptr to point to the memory right after ptr (total_size bytes from ptr)
    uint8_t *copy_ptr = reinterpret_cast<uint8_t*>(ptr) + total_size;
    
    test_put_get_notify_wait(stream, (uint8_t *)ptr, copy_ptr, my_pe, n_pes);
    std::cout << "[TEST] begin to exit...... my_pe: " << my_pe << std::endl;
    
    // Free only ptr, as copy_ptr points to a part of ptr's memory
    aclshmem_free(ptr);
    test_finalize(stream, device_id);
}

TEST(TEST_SYNC_API, TestShmemNotifyWait)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    // test_mutil_task(test_aclshmem_notify_wait, local_mem_size, process_count);
}