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

using namespace std;

class HostPutMemStreamTest {
public:
    inline HostPutMemStreamTest()
    {
    }
    inline void Init(uint8_t *gva, uint8_t *dev, int32_t rank_size_, size_t element_size_, aclrtStream st)
    {
        gva_gm = gva;
        dev_gm = dev;

        rank_size = rank_size_;
        element_size = element_size_;
        stream = st;
    }
    inline void Process()
    {
        int chunk_size = 16;
        for (int i = 0; i < rank_size; i++) {
            aclshmemx_putmem_on_stream(static_cast<void *>(gva_gm), static_cast<void *>(dev_gm),
                chunk_size, i % rank_size, stream);
        }
    }

private:
    uint8_t *gva_gm;
    uint8_t *dev_gm;

    int32_t rank_size;
    size_t element_size;
    aclrtStream stream;
};

void host_test_putmem_stream(uint8_t *gva, uint8_t *dev, int32_t rank_, size_t element_size_, aclrtStream stream)
{
    HostPutMemStreamTest op;
    op.Init(gva, dev, rank_, element_size_, stream);
    op.Process();
}

static void host_test_put_get_mem_stream(int rank_id, int rank_size, uint64_t local_mem_size, aclrtStream stream)
{
    int sleep_time = 1;
    int stage_total = 16;
    int stage_offset = 10;
    size_t ele_size = 16;
    int total_size = 16 * rank_size;
    size_t input_size = total_size;

    std::vector<uint8_t> input(total_size, 0);
    std::vector<uint8_t> out(total_size, 0);
    for (int i = 0; i < stage_total; i++) {
        input[i] = (rank_id + stage_offset);
    }

    void *dev_ptr = aclshmem_malloc(input_size);

    EXPECT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    void *ptr = aclshmem_malloc(1024);
    host_test_putmem_stream((uint8_t *)ptr + stage_total * rank_id, (uint8_t *)dev_ptr, rank_size, ele_size, stream);
    aclshmem_handle_t handle;
    handle.team_id = ACLSHMEM_TEAM_WORLD;
    aclshmemx_handle_wait(handle, stream);
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    sleep(sleep_time);

    EXPECT_EQ(aclrtMemcpy(out.data(), total_size, ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    string p_name = "[Process " + to_string(rank_id) + "] ";
    std::cout << "After putmemstream:" << p_name;
    for (int i = 0; i < total_size; i++) {
        std::cout << static_cast<int>(out[i]) << " ";
        int stage = i / stage_total;
        EXPECT_EQ(static_cast<int>(out[i]), stage + stage_offset);
    }
    std::cout << std::endl;
}

void test_host_aclshmem_putmem_stream(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_cross_init(rank_id, n_ranks, local_mem_size, &stream);
    EXPECT_NE(stream, nullptr);

    host_test_put_get_mem_stream(rank_id, n_ranks, local_mem_size, stream);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
}

TEST(TestPutMemStreamHostApi, TestShmemMemPutMemStream)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            test_host_aclshmem_putmem_stream(rank_id, n_ranks, local_mem_size);
        },
        local_mem_size, process_count);
}
