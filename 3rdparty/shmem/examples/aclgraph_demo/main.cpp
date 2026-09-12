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

#include "utils.h"
#include "param.h"

#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "shmem.h"
#include "aclgraph_demo_kernel.h"

int g_npus = 8;
const char *ipport = "tcp://127.0.0.1:8998";
int f_pe = 0;
int f_npu = 0;
const char *data_type = "int";
int loop_times = 3;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr uint32_t MAGIC_MULTIPLIER = 1024;
constexpr uint32_t DATA_SIZE_THRESHOLD = 2097152;
constexpr uint32_t BLOCK_NUM_SMALL_DATA = 8;
constexpr uint32_t BLOCK_NUM_LARGE_DATA = 16;

aclshmemx_uniqueid_t default_flag_uid;

template <class T>
void run_vector_add(int64_t numElements, void *a, void *b, void *c, void *stream);

template <class T>
int test_aclshmem_all_gather(int pe_id, int n_pes)
{
    // ACLStream init
    int status = 0;
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);

    // Prepare FFTS address
    uint64_t ffts_addr = util_get_ffts_config();

    int case_num = 2;
    std::vector<uint32_t> test_cases = {};
    for (int i = 0; i < case_num; i++) {
        int data_len = 262144 * (1 << i);
        test_cases.push_back(data_len);
    }

    uint32_t BLOCK_NUM = 8;

    // magic is used to sync.
    int magic = 1;

    for (int i = 0; i < test_cases.size(); i++) {

        aclmdlRI model = nullptr;

        if (pe_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Started." << std::endl;
        }
        uint32_t trans_size = test_cases[i];

        //  Small data kernel needs 8 AIV core, Big data kernel needs 16 AIV.
        if (trans_size * sizeof(T) < DATA_SIZE_THRESHOLD) {
            BLOCK_NUM = BLOCK_NUM_SMALL_DATA;
        } else {
            BLOCK_NUM = BLOCK_NUM_LARGE_DATA;
        }

        void *input_ptr;
        aclrtMalloc(&input_ptr, trans_size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        uint8_t *input_host;
        aclrtMallocHost(reinterpret_cast<void **>(&input_host), trans_size * sizeof(T));
        std::string inputFile = "../../examples/aclgraph_demo/golden/allgather_" + std::to_string(trans_size) + "_" +
                                std::to_string(n_pes) + "/input_gm_" + std::to_string(pe_id) + ".bin";
        ReadFile(inputFile, input_host, trans_size * sizeof(T));
        aclrtMemcpy(input_ptr, trans_size * sizeof(T), input_host, trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

        void *input_b_ptr;
        aclrtMalloc(&input_b_ptr, trans_size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(input_b_ptr, trans_size * sizeof(T), input_host, trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

        void *input_c_ptr;
        aclrtMalloc(&input_c_ptr, trans_size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

        void *output_ptr;
        aclrtMalloc(&output_ptr, trans_size * n_pes * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

        void *output_b_ptr;
        aclrtMalloc(&output_b_ptr, trans_size * n_pes * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

        void *output_c_ptr;
        aclrtMalloc(&output_c_ptr, trans_size * n_pes * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

        // sync Buffer + data Buffer
        int aiv_num = BLOCK_NUM;
        void *ptr = aclshmem_malloc(aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T));

        // magicNum
        int *magic_host_ptr;
        aclrtMallocHost(reinterpret_cast<void **>(&magic_host_ptr), sizeof(int));
        int magic_value;
        int magic_value_2;
        int *magic_ptr;
        aclrtMalloc(reinterpret_cast<void **>(&magic_ptr), sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
        int *magic_ptr2;
        aclrtMalloc(reinterpret_cast<void **>(&magic_ptr2), sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);

        // golden
        T *output_host;
        size_t output_size = n_pes * trans_size * sizeof(T);
        status = aclrtMallocHost(reinterpret_cast<void **>(&output_host), output_size);

        T *golden_host;
        status = aclrtMallocHost(reinterpret_cast<void **>(&golden_host), output_size);
        std::string goldenFile = "../../examples/aclgraph_demo/golden/allgather_" +
            std::to_string(trans_size) + "_" + std::to_string(n_pes) + "/golden.bin";
        ReadFile(goldenFile, golden_host, n_pes * trans_size * sizeof(T));

        // aclGraph(add + allgather + allgather + add)
        for (int zz = 0; zz < loop_times; zz++) {
            magic++;
            magic_value = magic * MAGIC_MULTIPLIER;
            magic_value_2 = (magic + loop_times + 10) * MAGIC_MULTIPLIER;
            *magic_host_ptr = magic_value;
            aclrtMemcpy(magic_ptr, sizeof(int), magic_host_ptr, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE);
            *magic_host_ptr = magic_value_2;
            aclrtMemcpy(magic_ptr2, sizeof(int), magic_host_ptr, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE);

            // capture
            if (zz == 0) {
                aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
                run_vector_add<T>(trans_size, input_ptr, input_b_ptr, input_c_ptr, stream);
                aclrtMemcpyAsync(input_ptr, trans_size * sizeof(T), input_c_ptr, trans_size * sizeof(T),
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
                allgather_demo<T>(BLOCK_NUM, stream, ffts_addr, (uint8_t *)input_ptr,
                              (uint8_t *)output_ptr, (uint8_t *)ptr, trans_size, magic_ptr);
                allgather_demo<T>(BLOCK_NUM, stream, ffts_addr, (uint8_t *)input_b_ptr,
                              (uint8_t *)output_b_ptr, (uint8_t *)ptr, trans_size, magic_ptr2);
                run_vector_add<T>(n_pes * trans_size, output_ptr, output_b_ptr, output_c_ptr, stream);
                aclmdlRICaptureEnd(stream, &model);
                aclmdlRIExecuteAsync(model, stream);

            // replay
            } else {
                aclmdlRIExecuteAsync(model, stream);
            }

            status = aclrtSynchronizeStream(stream);
            if (status != 0) {
                std::cout << "sync stream failed\n";
            }

            aclrtMemcpy(output_host, output_size, output_c_ptr, output_size, ACL_MEMCPY_DEVICE_TO_HOST);

            // Check aclGraph
            for (int zzz = 0; zzz < n_pes * trans_size; zzz++) {
                if (static_cast<int>(output_host[zzz]) != static_cast<int>(golden_host[zzz]) * (zz + 3)) {
                    std::cout << "ERROR each loop: " << static_cast<int>(output_host[zzz]) << " != "
                              << static_cast<int>(golden_host[zzz]) * (zz + 3)
                              << ", trans_size: " << trans_size << ", idx is: " << zzz
                              << ", pe_id is: " << pe_id << std::endl;
                    status = -1;
                    break;
                }
            }
        }

        aclshmemx_show_prof(nullptr, true);


        // 去初始化
        aclrtFreeHost(input_host);
        aclrtFreeHost(magic_host_ptr);
        aclrtFreeHost(output_host);
        aclrtFreeHost(golden_host);

        aclshmem_free(ptr);
        aclrtFree(input_ptr);
        aclrtFree(input_b_ptr);
        aclrtFree(input_c_ptr);
        aclrtFree(output_ptr);
        aclrtFree(output_b_ptr);
        aclrtFree(output_c_ptr);
        aclrtFree(magic_ptr);
        aclrtFree(magic_ptr2);

        if (pe_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Finished !! Result Correct !!" << std::endl;
        }

        aclmdlRIDestroy(model);
    }

    aclrtDestroyStream(stream);
    return status;
}

int main(int argc, char *argv[])
{
    if (argc != INDEX9) {
        std::cerr << "Usage: " << argv[0]
                  << " <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> <data_type> <loop_times>" << std::endl;
        return EXIT_FAILURE;
    }

    int status = 0;
    int n_pes = atoi(argv[INDEX1]);
    int pe_id = atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    g_npus = atoi(argv[INDEX4]);
    f_pe = atoi(argv[INDEX5]);
    f_npu = atoi(argv[INDEX6]);
    data_type = argv[INDEX7];
    loop_times = atoi(argv[INDEX8]);

    // Acl && Shmem init
    int32_t device_id = pe_id % g_npus + f_npu;
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    if (std::string(data_type) == "int") {
        status = test_aclshmem_all_gather<int>(pe_id, n_pes);
    } else {
        std::cerr << "only support data_type == int" << std::endl;
        return EXIT_FAILURE;
    }
    status = aclshmem_finalize();
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    return 0;
}
