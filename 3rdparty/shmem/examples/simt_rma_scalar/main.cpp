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

#include "shmem.h"
#include "acl/acl.h"
#include "kernel_operator.h"
#include "../utils/utils.h"


const char *ipport = "tcp://127.0.0.1:8998";

aclshmemx_uniqueid_t default_flag_uid;

__simt_vf__ __launch_bounds__(1024) inline void demo_call_simt(
    __gm__ int32_t* sym_input,
    __gm__ int32_t* output, 
    __gm__ uint64_t* dbg
) 
{
    int32_t mype = simt::aclshmem_my_pe();
    int32_t npes = simt::aclshmem_n_pes();

    int32_t peer = (mype + 1) % npes;
    simt::aclshmem_int32_p(sym_input, peer, peer);
    auto get_num = simt::aclshmem_int32_g(sym_input, peer);
    *output = get_num;
}

__global__ __vector__ void demo_call(__gm__ int* input, __gm__ int* output, __gm__ uint64_t* dbg)
{
    asc_vf_call<demo_call_simt>(dim3(1), input, output, dbg);
}

void run_demo_scalar(uint32_t block_dim, void* stream, int* input, int* output, uint64_t* dbg)
{
    demo_call<<<1, 0, stream>>>(
        input, output, dbg
    );
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
    *input_host = -1;
    *output_host = -1;

    uint64_t* debug_host;
    constexpr int32_t debug_size = 32;
    ACL_CHECK_WITH_RET(aclrtMallocHost(reinterpret_cast<void**>(&debug_host), sizeof(uint64_t) * debug_size),
        ERROR_LOG("aclrtMallocHost failed"), return -1);
    for (int i = 0;i < debug_size;i++) {
        debug_host[i] = 0;
    }

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(my_pe, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    auto status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("aclshmemx_init_attr failed"), return -1);

    uint8_t *input_device = (uint8_t*)aclshmem_malloc(2*1024*1024);
    uint8_t *output_device = nullptr;
    ACL_CHECK_WITH_RET(aclrtMalloc((void **)&output_device, sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
        ERROR_LOG("aclrtMalloc failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMemcpy(input_device, sizeof(int), input_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("aclrtMemcpy failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMemcpy(output_device, sizeof(int), output_host, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("aclrtMemcpy failed"), return -1);

    uint64_t* debug_device = nullptr;
    ACL_CHECK_WITH_RET(aclrtMalloc((void **)&debug_device, sizeof(uint64_t) * debug_size, ACL_MEM_MALLOC_HUGE_FIRST),
        ERROR_LOG("aclrtMalloc failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMemcpy(debug_device, sizeof(uint64_t) * debug_size, debug_host, sizeof(uint64_t) * debug_size, ACL_MEMCPY_HOST_TO_DEVICE),
        ERROR_LOG("aclrtMemcpy failed"), return -1);

    aclshmem_barrier_all();
    run_demo_scalar(1, stream, (int*)input_device, (int*)output_device, debug_device);

    int32_t ret = aclrtSynchronizeStream(stream);
    if (ret != 0) {
        ERROR_LOG("aclrtSynchronizeStream failed, error code: %d", ret);
        return -1;
    }
    aclshmem_barrier_all();

    ACL_CHECK_WITH_RET(aclrtMemcpy(input_host, sizeof(int), input_device, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("aclrtMemcpy failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMemcpy(output_host, sizeof(int), output_device, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("aclrtMemcpy failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtMemcpy(debug_host, sizeof(uint64_t) * debug_size, debug_device, sizeof(uint64_t) * debug_size, ACL_MEMCPY_DEVICE_TO_HOST),
        ERROR_LOG("aclrtMemcpy failed"), return -1);

    printf("%d: received message %d %d\n", my_pe, *input_host, *output_host);
    if ((*output_host == ((my_pe + 1) % n_pes)) && (*input_host == my_pe)) {
        printf("[SUCCESS] run success in pe %d\n", my_pe);
    } else {
        printf("[ERROR] run result incorrect in pe %d, hopes %d, given %d\n", my_pe, ((my_pe + 1) % n_pes), *output_host);
    }

    std::cout << "Debug Info: #[";
    for (int i = 0; i < debug_size; i++) {
        std::cout << debug_host[i] << ", ";
    }
    std::cout << "]" << std::endl;

    aclshmem_free(input_device);
    aclshmem_finalize();
    ACL_CHECK_WITH_RET(aclrtFreeHost(input_host), ERROR_LOG("aclrtFreeHost failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtFreeHost(output_host), ERROR_LOG("aclrtFreeHost failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtFreeHost(debug_host), ERROR_LOG("aclrtFreeHost failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtFree(output_device), ERROR_LOG("aclrtFree failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtFree(debug_device), ERROR_LOG("aclrtFree failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtDestroyStream(stream), ERROR_LOG("aclrtDestroyStream failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtResetDevice(my_pe), ERROR_LOG("aclrtResetDevice failed"), return -1);
    ACL_CHECK_WITH_RET(aclFinalize(), ERROR_LOG("aclFinalize failed"), return -1);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        ERROR_LOG("Usage: %s <n_pes> <my_pe>", argv[0]);
        return -1;
    }

    int n_pes = atoi(argv[1]);
    int my_pe = atoi(argv[2]);

    (void)test_aclshmem_rma_scalar_8p(my_pe, n_pes);
    INFO_LOG("[INFO] demo run end in pe %d.", my_pe);
    return 0;
}
