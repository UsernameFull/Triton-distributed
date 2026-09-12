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
const char* ipport;
int f_pe = 0;
int f_npu = 0;
constexpr uint64_t DEBUG_DUMP_SIZE = 200 * 1024 * 1024;
extern void launch_udma_all_gather(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dump, int message_length);
extern void launch_udma_put_signal(
    uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr, uint8_t* dump_addr, int elements,
    uint64_t signal);

aclshmemx_uniqueid_t default_flag_uid;

// Common initialization function
int init_acl_shmem(
    int pe_id, int n_pes, uint64_t local_mem_size, int32_t& device_id, aclrtStream& stream, uint8_t*& ptr)
{
    int status = 0;
    device_id = pe_id % g_npus + f_npu;

    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    status |= aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    ptr = static_cast<uint8_t*>(aclshmem_malloc(1024));
    return status;
}

// Common data initialization function
void init_data(int pe_id, uint8_t* ptr, uint32_t trans_size)
{
    const int num10 = 10;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (pe_id + num10);
    }

    aclrtMemcpy(
        ptr + aclshmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t), input.data(),
        trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
}

// Common result validation function
bool validate_result(uint8_t* ptr, int n_pes, uint32_t trans_size)
{
    const int num10 = 10;
    int32_t* y_host;
    size_t input_size = n_pes * trans_size * sizeof(int32_t);
    aclrtMallocHost(reinterpret_cast<void**>(&y_host), input_size);
    aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST);

    const int block_size = 16;
    bool success = true;
    for (int i = 0; i < n_pes; i++) {
        for (int j = 0; j < block_size; j++) {
            if (y_host[trans_size * i + trans_size / block_size * j] != num10 + i) {
                std::cout << y_host[trans_size * i + trans_size / block_size * j] << " != " << num10 + i << std::endl;
                success = false;
            }
        }
    }

    aclrtFreeHost(y_host);
    return success;
}

// Common cleanup function
int cleanup_resources(aclrtStream stream, int32_t device_id, uint8_t* ptr, uint8_t* extra_ptr = nullptr)
{
    int status = 0;
    aclshmem_free(ptr);
    if (extra_ptr != nullptr) {
        aclshmem_free(extra_ptr);
    }
    status |= aclshmem_finalize();
    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();
    return status;
}

int test_aclshmem_team_all_gather(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int status = 0;
    int32_t device_id;
    aclrtStream stream = nullptr;
    uint8_t* ptr = nullptr;

    status = init_acl_shmem(pe_id, n_pes, local_mem_size, device_id, stream, ptr);
    if (status != 0) {
        return status;
    }

    uint32_t trans_size = 16;
    init_data(pe_id, ptr, trans_size);

    uint8_t* dump = nullptr;
#if defined(ENABLE_ASCENDC_DUMP)
    (void)aclrtMalloc((void**)&dump, DEBUG_DUMP_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (dump == nullptr) {
        std::cout << "dump workspace is nullptr" << std::endl;
        return -1;
    }
#endif
    // Launch the all-gather kernel.
    launch_udma_all_gather(1, stream, (uint8_t*)ptr, dump, trans_size * sizeof(int32_t));
    status |= aclrtSynchronizeStream(stream);
#if defined(ENABLE_ASCENDC_DUMP)
    Adx::AdumpPrintWorkSpace(dump, DEBUG_DUMP_SIZE, stream, "udma_demo");
#endif

    if (validate_result(ptr, n_pes, trans_size)) {
        std::cout << "check transport result success, pe=" << pe_id << std::endl;
    } else {
        std::cout << "check transport result failed, pe=" << pe_id << std::endl;
        cleanup_resources(stream, device_id, ptr);
        return -1;
    }

    return cleanup_resources(stream, device_id, ptr);
}

int test_aclshmem_udma_put_signal(int pe_id, int n_pes, uint64_t local_mem_size)
{
    int status = 0;
    int32_t device_id;
    aclrtStream stream = nullptr;
    uint8_t* ptr = nullptr;

    status = init_acl_shmem(pe_id, n_pes, local_mem_size, device_id, stream, ptr);
    if (status != 0) {
        return status;
    }

    // Allocate signal address space for all PEs
    uint8_t* sig_addr = static_cast<uint8_t*>(aclshmem_malloc(n_pes * sizeof(uint64_t)));
    // Initialize signal addresses to 0 to avoid dirty data
    std::vector<uint64_t> init_signals(n_pes, 0);
    aclrtMemcpy(
        sig_addr, n_pes * sizeof(uint64_t), init_signals.data(), n_pes * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // Allocate dump workspace
    uint8_t* dump = nullptr;
#if defined(ENABLE_ASCENDC_DUMP)
    (void)aclrtMalloc((void**)&dump, DEBUG_DUMP_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (dump == nullptr) {
        std::cout << "dump workspace is nullptr" << std::endl;
        return -1;
    }
#endif

    uint32_t trans_size = 16;
    init_data(pe_id, ptr, trans_size);

    // Launch the put signal kernel.
    uint64_t signal = 1000;
    launch_udma_put_signal(1, stream, (uint8_t*)ptr, sig_addr, dump, trans_size * sizeof(int32_t), signal);
    status |= aclrtSynchronizeStream(stream);

    if (validate_result(ptr, n_pes, trans_size)) {
        std::cout << "check udma put signal result success, pe=" << pe_id << std::endl;
    } else {
        std::cout << "check udma put signal result failed, pe=" << pe_id << std::endl;
        cleanup_resources(stream, device_id, ptr, sig_addr);
        return -1;
    }

    // Read and validate all signals
    std::vector<uint64_t> signal_values(n_pes, 0);
    status |= aclrtMemcpy(
        signal_values.data(), n_pes * sizeof(uint64_t), sig_addr, n_pes * sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "Signal values received on pe=" << pe_id << ":" << std::endl;
    bool all_signals_set = true;
    for (int i = 0; i < n_pes; i++) {
        std::cout << "  PE " << i << ": " << signal_values[i] << std::endl;
        if (i == pe_id) {
            continue;
        }
        if (signal_values[i] != signal) {
            all_signals_set = false;
        }
    }

    // Check if all signals are set
    if (!all_signals_set) {
        std::cout << "[ERROR]: Some signals not equal " << signal << ", may not have been set properly" << std::endl;
        return -1;
    } else {
        std::cout << "All signals are set successfully" << std::endl;
    }

    // Free dump workspace
#if defined(ENABLE_ASCENDC_DUMP)
    if (dump != nullptr) {
        aclrtFree(dump);
    }
#endif
    return cleanup_resources(stream, device_id, ptr, sig_addr);
}

int main(int argc, char* argv[])
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

    // Default to run all-gather test if no test type specified
    int test_type = 0;
    if (argIdx < argc) {
        test_type = atoi(argv[argIdx++]);
    }

    if (test_type == 1) {
        status = test_aclshmem_udma_put_signal(pe_id, n_pes, local_mem_size);
    } else {
        status = test_aclshmem_team_all_gather(pe_id, n_pes, local_mem_size);
    }

    if (status != 0) {
        std::cout << "[FAILED] demo run failed in pe " << pe_id << ", status=" << status << std::endl;
    } else {
        std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    }
    return status;
}
