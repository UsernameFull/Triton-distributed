/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cstring>
#include <cerrno>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "shmem.h"
#include "../utils/utils.h"


const char *ipport = "tcp://127.0.0.1:8998";


aclshmemx_uniqueid_t default_flag_uid;

extern "C" __global__ __aicore__ void kernel_test(__gm__ int* input, __gm__ int* output)
{
    if ASCEND_IS_AIC {
        return;
    }

    auto coreId = AscendC::GetBlockIdx();
    if (coreId > 0) {
        return;
    }

    int mype = aclshmem_my_pe();
    int npes = aclshmem_n_pes();
    int peer = (mype + 1) % npes;

    for (int iii = 0; iii < 10; iii++) {
        aclshmem_int32_p(input, peer, peer);
        aclshmem_quiet();
        auto get_num = aclshmem_int32_g(input, peer);
        aclshmem_quiet();
        *(output ) = get_num;
    }
}

void run_demo_scalar(uint32_t block_dim, void* stream, int* input, int* output)
{
    kernel_test<<<block_dim, nullptr, stream>>>(input, output);
}

int test_aclshmem_rma_scalar_8p(int my_pe, int n_pes)
{
    // 初始化ACL和ACLSHMEM
    aclrtStream stream = nullptr;

    ACL_CHECK_WITH_RET(aclInit(nullptr), ERROR_LOG("aclInit failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtSetDevice(my_pe), ERROR_LOG("aclrtSetDevice failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtCreateStream(&stream), ERROR_LOG("aclrtCreateStream failed"), return -1);

    int32_t *input_host;
    int32_t *output_host;
    ACL_CHECK_WITH_RET(aclrtMallocHost(reinterpret_cast<void**>(&input_host), sizeof(int)),
        ERROR_LOG("aclrtMallocHost failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMallocHost(reinterpret_cast<void**>(&output_host), sizeof(int)),
        ERROR_LOG("aclrtMallocHost failed"), return -1);
    *input_host = 0;
    *output_host = my_pe;

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(my_pe, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    auto status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("aclshmemx_init_attr failed"), return -1);

    uint8_t *input = (uint8_t*)aclshmemx_malloc(2*1024*1024, HOST_SIDE);
    uint8_t *output = nullptr;
    ACL_CHECK_WITH_RET(aclrtMalloc((void **)&output, sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
        ERROR_LOG("aclrtMalloc failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMemcpy(input, sizeof(int), input_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("aclrtMemcpy failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMemcpy(output, sizeof(int), output_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("aclrtMemcpy failed"), return -1);

    aclshmem_barrier_all();
    run_demo_scalar(1, stream, (int*)input, (int*)output);

    ACL_CHECK_WITH_RET(aclrtSynchronizeStream(stream), ERROR_LOG("aclrtSynchronizeStream failed"), return -1);
    aclshmem_barrier_all();

    ACL_CHECK_WITH_RET(aclrtMemcpy(input_host, sizeof(int), input, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("aclrtMemcpy failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMemcpy(output_host, sizeof(int), output, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("aclrtMemcpy failed"), return -1);

    printf("%d: received message %d %d\n", my_pe, *input_host, *output_host);
    if ( *output_host == ((my_pe + 1) % n_pes)) {
        printf("[SUCCESS] run success in pe %d\n", my_pe);
    } else {
        printf("[ERROR] run result incorrect in pe %d\n", my_pe);  // 期望input变为前卡, output变为后卡
    }

    aclshmemx_free(input, HOST_SIDE);
    aclshmem_finalize();
    ACL_CHECK_WITH_RET(aclrtFreeHost(input_host), ERROR_LOG("aclrtFreeHost failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtFreeHost(output_host), ERROR_LOG("aclrtFreeHost failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtDestroyStream(stream), ERROR_LOG("aclrtDestroyStream failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtResetDevice(my_pe), ERROR_LOG("aclrtResetDevice failed"), return -1);
    ACL_CHECK_WITH_RET(aclFinalize(), ERROR_LOG("aclFinalize failed"), return -1);
    return 0;
}

int main(int argc, char *argv[])
{
    int argIdx = 1;
    int n_pes = atoi(argv[argIdx++]);
    int my_pe = atoi(argv[argIdx++]);

    (void)test_aclshmem_rma_scalar_8p(my_pe, n_pes);
    INFO_LOG("[INFO] demo run end in pe %d.", my_pe);
    return 0;
}
