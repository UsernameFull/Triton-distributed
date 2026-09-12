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
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "shmem.h"
#include "utils.h"
#include "param.h"
#include "rdma_aclgraph_demo_kernel.h"

int g_npus = 8;
const char *ipport;
int f_pe = 0;
int f_npu = 0;

aclshmemx_uniqueid_t default_flag_uid;

template <class T>
void run_vector_add(int64_t numElements, void *a, void *b, void *c, void *stream);

constexpr uint32_t TRANS_SIZE = 16;
constexpr size_t SYMMETRIC_MEM_SIZE = 1024;
constexpr int NUM10 = 10;
constexpr int LOOP_TIMES = 3;

int test_aclshmem_team_all_gather(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;

    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);
    if (status != 0) {
        std::cout << "create device env failed\n";
        return -1;
    }

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint8_t *ptr = static_cast<uint8_t *>(aclshmem_malloc(SYMMETRIC_MEM_SIZE));
    uint8_t *b_ptr = static_cast<uint8_t *>(aclshmem_malloc(SYMMETRIC_MEM_SIZE));

    std::vector<int32_t> input(TRANS_SIZE, 0);
    for (uint32_t i = 0; i < TRANS_SIZE; i++) {
        input[i] = (pe_id + NUM10);
    }

    status |= aclrtMemcpy(ptr + aclshmem_my_pe() * TRANS_SIZE * sizeof(int32_t),
                TRANS_SIZE * sizeof(int32_t), input.data(), TRANS_SIZE * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
    status |= aclrtMemcpy(b_ptr + aclshmem_my_pe() * TRANS_SIZE * sizeof(int32_t),
                TRANS_SIZE * sizeof(int32_t), input.data(), TRANS_SIZE * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    if (status != 0) {
        std::cout << "init rdma ptr failed\n";
        return -1;
    }

    size_t input_size = n_pes * TRANS_SIZE * sizeof(int32_t);
    void *c_ptr;
    aclrtMalloc(&c_ptr, input_size, ACL_MEM_MALLOC_HUGE_FIRST);

    void *add_ptr;
    aclrtMalloc(&add_ptr, input_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(static_cast<uint8_t *>(add_ptr) + aclshmem_my_pe() * TRANS_SIZE * sizeof(int32_t), TRANS_SIZE * sizeof(int32_t), input.data(),
                TRANS_SIZE * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    void *temp_ptr;
    aclrtMalloc(&temp_ptr, input_size, ACL_MEM_MALLOC_HUGE_FIRST);

    int32_t *y_host;
    aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size);

    aclmdlRI model = nullptr;
    aclshmem_handle_t handle;
    handle.team_id = ACLSHMEM_TEAM_WORLD;

    for (int zz = 0; zz < LOOP_TIMES; zz++) {
        if (zz == 0) {
            aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
            run_vector_add<int>(n_pes * TRANS_SIZE, ptr, add_ptr, temp_ptr, stream);
            aclrtMemcpyAsync(ptr, input_size, temp_ptr, input_size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            aclshmemx_handle_wait(handle, stream);
            allgather_demo(1, stream, (uint8_t *)ptr, TRANS_SIZE * sizeof(int32_t));
            aclshmemx_handle_wait(handle, stream);
            allgather_demo(1, stream, (uint8_t *)b_ptr, TRANS_SIZE * sizeof(int32_t));
            aclshmemx_handle_wait(handle, stream);
            run_vector_add<int>(n_pes * TRANS_SIZE, ptr, b_ptr, c_ptr, stream);
            aclmdlRICaptureEnd(stream, &model);
            aclmdlRIExecuteAsync(model, stream);
        } else {
            aclmdlRIExecuteAsync(model, stream);
        }

        aclrtMemcpyAsync(y_host, input_size, c_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
        status = aclrtSynchronizeStream(stream);
        if (status != 0) {
            std::cout << "sync stream failed\n";
        }
        for (int i = 0; i < n_pes; i++) {
            for (uint32_t j = 0; j < TRANS_SIZE; j++) {
                int32_t actual = y_host[TRANS_SIZE * i + j];
                int32_t expected = (NUM10 + i) * (3 + zz);
                if (actual != expected) {
                    std::cout << "ERROR loop: " << zz << ", pe: " << aclshmem_my_pe()
                              << ", pe_id: " << i << ", idx: " << TRANS_SIZE * i + j
                              << ", actual: " << actual << " != expected: " << expected << std::endl;
                    status = -1;
                    break;
                }
            }
        }
    }

    std::cout << "check transport result success, relative pe=" << pe_id << std::endl;

    aclrtFreeHost(y_host);
    aclshmem_free(ptr);
    aclshmem_free(b_ptr);
    aclrtFree(c_ptr);
    aclrtFree(add_ptr);
    aclrtFree(temp_ptr);

    aclmdlRIDestroy(model);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return status;
}

int main(int argc, char *argv[])
{
    if (argc != INDEX7) {
        std::cerr << "Usage: " << argv[0]
                  << " <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>" << std::endl;
        return EXIT_FAILURE;
    }

    int argIdx = 1;
    int status = 0;
    int n_pes = atoi(argv[argIdx++]);
    int pe_id = atoi(argv[argIdx++]);
    ipport = argv[argIdx++];
    g_npus = atoi(argv[argIdx++]);
    f_pe = atoi(argv[argIdx++]);
    f_npu = atoi(argv[argIdx++]);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    status = test_aclshmem_team_all_gather(pe_id, n_pes, local_mem_size);

    if (status) {
        std::cerr << "[FAILED] demo run failed in relative pe " << pe_id << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in relative pe " << pe_id << std::endl;
    return 0;
}
