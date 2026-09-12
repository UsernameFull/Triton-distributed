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
#include <string>
#include <vector>
#include <algorithm>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "utils.h"
#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

int g_npus = 8;
const char *ipport;
int f_pe = 0;
int f_npu = 0;
constexpr uint64_t DEBUG_DUMP_SIZE = 200 * 1024 * 1024;
extern void launch_udma_atomic_add(uint32_t block_dim, void *stream, uint8_t *gva, uint8_t *dump, int message_length);

aclshmemx_uniqueid_t default_flag_uid;

int test_aclshmem_udma_atomic_add(int pe_id, int n_pes, uint64_t local_mem_size)
{
    // Initialize ACL and ACLSHMEM.
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    const int num10 = 10;
    aclrtStream stream = nullptr;

    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    status |= aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint8_t *ptr = static_cast<uint8_t *>(aclshmem_malloc(1024));

    // Initialize input data.
    uint32_t trans_size = 1;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = pe_id + num10;
    }

    status |= aclrtMemcpy(ptr, trans_size * sizeof(int32_t), input.data(), trans_size * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    uint8_t *dump = nullptr;
#if defined(ENABLE_ASCENDC_DUMP)
    (void)aclrtMalloc((void **)&dump, DEBUG_DUMP_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (dump == nullptr) {
        std::cout << "dump workspace is nullptr" << std::endl;
        return -1;
    }
#endif
    // Launch the udma atomic add kernel.
    launch_udma_atomic_add(1, stream, (uint8_t *)ptr, dump, trans_size * sizeof(int32_t));
    status |= aclrtSynchronizeStream(stream);
    aclshmem_barrier_all();
#if defined(ENABLE_ASCENDC_DUMP)
    Adx::AdumpPrintWorkSpace(dump, DEBUG_DUMP_SIZE, stream, "udma_atomic_add");
#endif
    // Copy back and validate the result.
    int32_t *y_host;
    size_t input_size = trans_size * sizeof(int32_t);
    status |= aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size);
    status |= aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST);

    if (y_host[0] != (pe_id + num10 * 2)) {
        std::cout << "pe" << pe_id << ": " << y_host[0] << " != " << (pe_id + num10 * 2) << std::endl;
        status |= -1;
    } else {
        std::cout << "check transport result success, pe=" << pe_id << std::endl;
    }

    // Release resources.
    status |= aclrtFreeHost(y_host);
    aclshmem_free(ptr);
    status |= aclshmem_finalize();
    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();
    return status;
}

int main(int argc, char *argv[])
{
    int argIdx = 1;
    int status = 0;
    int n_pes = atoi(argv[argIdx++]);
    int pe_id = atoi(argv[argIdx++]);
    ipport = argv[argIdx++];
    g_npus = atoi(argv[argIdx++]);
    f_pe = atoi(argv[argIdx++]);
    f_npu = atoi(argv[argIdx++]);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    status = test_aclshmem_udma_atomic_add(pe_id, n_pes, local_mem_size);
    if (status != 0) {
        std::cout << "[FAILED] demo run failed in pe " << pe_id << ", status=" << status << std::endl;
    } else {
        std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    }
    return status;
}
