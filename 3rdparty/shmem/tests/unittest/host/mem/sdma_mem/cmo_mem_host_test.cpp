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
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern int32_t test_sdma_init(int pe, int npes, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

extern void copy_perf_kernel(uint32_t block_dim, void* stream, 
                                uint8_t* src, uint8_t* res, uint32_t copypad_size, uint32_t copypad_times);
extern void cmo_pretech_kernel(uint8_t* src, uint32_t size, void* stream);

static void test_cmo_function(aclrtStream stream, uint32_t pe, uint32_t npes)
{
    uint32_t n_blocks = 20;
    uint32_t aiv_num = n_blocks * 2;
    uint32_t copypad_size = 64 * 1024;
    uint32_t copypad_times = 32;

    uint32_t src_size = aiv_num * copypad_size * copypad_times;
    uint32_t res_size = sizeof(uint32_t) * aiv_num;

    void *src_ptr;
    void *res_ptr;
    void *res_host;

    uint32_t no_prefetch_cycles = 0;
    uint32_t host_prefetch_cycles = 0;
    uint32_t device_prefetch_cycles = 0;
    
    // no prefetch
    ASSERT_EQ(aclrtMalloc((void **) &(src_ptr), src_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);
    ASSERT_EQ(aclrtMalloc((void **) &(res_ptr), res_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);
    aclrtMallocHost((void **)(&res_host), res_size);
    copy_perf_kernel(n_blocks, stream, 
                        (uint8_t *)src_ptr, (uint8_t *)res_ptr, copypad_size, copypad_times);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclrtMemcpy(res_host, res_size, res_ptr, res_size, ACL_MEMCPY_DEVICE_TO_HOST);
    for (int32_t i = 0; i < aiv_num; i++)
    {
        no_prefetch_cycles += ((uint32_t*)res_host)[i];
    }
    std::cout << "[TEST] no_prefetch_cycles: " << no_prefetch_cycles << std::endl;
    ASSERT_EQ(aclrtFree(src_ptr), 0);
    ASSERT_EQ(aclrtFree(res_ptr), 0);
    ASSERT_EQ(aclrtFree(res_host), 0);

    // host prefetch
    ASSERT_EQ(aclrtMalloc((void **) &(src_ptr), src_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);
    ASSERT_EQ(aclrtMalloc((void **) &(res_ptr), res_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);
    aclrtMallocHost((void **)(&res_host), res_size);
    ASSERT_EQ(aclrtCmoAsync(src_ptr, src_size, ACL_RT_CMO_TYPE_PREFETCH, stream), 0);   // aclrtCmoAsync
    copy_perf_kernel(n_blocks, stream, 
                        (uint8_t *)src_ptr, (uint8_t *)res_ptr, copypad_size, copypad_times);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclrtMemcpy(res_host, res_size, res_ptr, res_size, ACL_MEMCPY_DEVICE_TO_HOST);
    for (int32_t i = 0; i < aiv_num; i++)
    {
        host_prefetch_cycles += ((uint32_t*)res_host)[i];
    }
    std::cout << "[TEST] host_prefetch_cycles: " << host_prefetch_cycles << std::endl;
    ASSERT_EQ(aclrtFree(src_ptr), 0);
    ASSERT_EQ(aclrtFree(res_ptr), 0);
    ASSERT_EQ(aclrtFree(res_host), 0);

    // device prefetch
    ASSERT_EQ(aclrtMalloc((void **) &(src_ptr), src_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);
    ASSERT_EQ(aclrtMalloc((void **) &(res_ptr), res_size, ACL_MEM_MALLOC_HUGE_FIRST), 0);
    aclrtMallocHost((void **)(&res_host), res_size);
    cmo_pretech_kernel((uint8_t *)src_ptr, src_size, stream);   // cmo_pretech_kernel
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    copy_perf_kernel(n_blocks, stream, 
                        (uint8_t *)src_ptr, (uint8_t *)res_ptr, copypad_size, copypad_times);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclrtMemcpy(res_host, res_size, res_ptr, res_size, ACL_MEMCPY_DEVICE_TO_HOST);
    for (int32_t i = 0; i < aiv_num; i++)
    {
        device_prefetch_cycles += ((uint32_t*)res_host)[i];
    }
    std::cout << "[TEST] device_prefetch_cycles: " << device_prefetch_cycles << std::endl;
    ASSERT_EQ(aclrtFree(src_ptr), 0);
    ASSERT_EQ(aclrtFree(res_ptr), 0);
    ASSERT_EQ(aclrtFree(res_host), 0);

    // copy_perf
    ASSERT_EQ(no_prefetch_cycles >= 2 * host_prefetch_cycles, true);
    ASSERT_EQ(no_prefetch_cycles >= 2 * device_prefetch_cycles, true);
}

void test_aclshmem_cmo_mem(int pe, int npes, uint64_t local_mem_size)
{
    int32_t device_id = pe % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_sdma_init(pe, npes, local_mem_size, &stream);
    if (status != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    test_cmo_function(stream, pe, npes);

    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemCMOMem)
{
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    return;
}