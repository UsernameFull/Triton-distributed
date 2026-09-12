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
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <iomanip>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <algorithm>

#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"
#include "utils.h"
#include "param.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

#include "acl/acl.h"
#include "shmem.h"
#include "allgather_kernel.h"

int g_npus = 8;
const char *ipport = "tcp://127.0.0.1:8998";
int f_pe = 0;
int f_npu = 0;
const char *data_type = "int";
int perf_times = 50;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr uint32_t MAGIC_MULTIPLIER = 1024;
constexpr uint32_t DATA_SIZE_THRESHOLD = 2097152;
constexpr uint32_t BLOCK_NUM_SMALL_DATA = 8;
constexpr uint32_t BLOCK_NUM_LARGE_DATA = 16;

aclshmemx_uniqueid_t default_flag_uid;

template <class T>
int test_aclshmem_all_gather(int pe_id, int n_pes)
{
    // ACLStream init
    int status = 0;
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);

    // Prepare FFTS address
    uint64_t fftsAddr = util_get_ffts_config();

    int case_num = 24;
    std::vector<uint32_t> test_cases = {};
    for (int i = 0; i < case_num; i++) {
        int data_len = 16 * (1 << i);
        test_cases.push_back(data_len);
    }

    uint32_t BLOCK_NUM = 8;

    std::string exec_path = __FILE__;
    size_t pos = exec_path.find_last_of("/\\");
    std::string dir_path = exec_path.substr(0, pos);
    std::string result_path = dir_path + "/results.csv";
    std::ofstream outFile(result_path);
    if (!outFile.is_open()) {
        std::cerr << "错误：无法创建文件！" << std::endl;
        return 1;
    }
    outFile << "M,N,Time(us)\n";

    // magic is used to sync.
    int magic = 1;

    for (int i = 0; i < test_cases.size(); i++) {
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
        aclrtMallocHost(reinterpret_cast<void**>(&input_host), trans_size * sizeof(T));
        std::string inputFile = "../../examples/allgather/golden/allgather_" + std::to_string(trans_size) + "_" +
                                std::to_string(n_pes) + "/input_gm_" + std::to_string(pe_id) + ".bin";
        ReadFile(inputFile, input_host, trans_size * sizeof(T));
        aclrtMemcpy(input_ptr, trans_size * sizeof(T), input_host, trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

        void *output_ptr;
        aclrtMalloc(&output_ptr, trans_size * n_pes * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

        // sync Buffer + data Buffer
        int aiv_num = BLOCK_NUM;
        void *ptr = aclshmem_malloc(aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T));

        // AllGather
        for (int zz = 0; zz < perf_times; zz++) {
            magic++;
            allgather_demo<T>(BLOCK_NUM, stream, fftsAddr, (uint8_t *)input_ptr,
                              (uint8_t *)output_ptr, (uint8_t *)ptr, trans_size, magic * MAGIC_MULTIPLIER);
        }
        status = aclrtSynchronizeStream(stream);

        aclshmemx_show_prof(nullptr, true);

        // Result Check
        T *output_host;
        size_t output_size = n_pes * trans_size * sizeof(T);
        status = aclrtMallocHost(reinterpret_cast<void**>(&output_host), output_size);
        status = aclrtMemcpy(output_host, output_size, output_ptr, output_size, ACL_MEMCPY_DEVICE_TO_HOST);

        T *golden_host;
        status = aclrtMallocHost(reinterpret_cast<void**>(&golden_host), output_size);
        std::string goldenFile = "../../examples/allgather/golden/allgather_" +
            std::to_string(trans_size) + "_" + std::to_string(n_pes) + "/golden.bin";
        ReadFile(goldenFile, golden_host, n_pes * trans_size * sizeof(T));
        for (int zz = 0; zz < n_pes * trans_size; zz++) {
            if (static_cast<float>(output_host[zz]) != static_cast<float>(golden_host[zz])) {
                std::cout << static_cast<float>(output_host[zz]) << " != " << static_cast<float>(golden_host[zz])
                          << ", trans_size is : " << trans_size << ", idx is: " << zz
                          << ", pe_id is: "<< pe_id << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // 去初始化
        status = aclrtFreeHost(input_host);
        status = aclrtFreeHost(output_host);
        status = aclrtFreeHost(golden_host);

        aclshmem_free(ptr);
        aclrtFree(input_ptr);
        aclrtFree(output_ptr);

        outFile << 1 << "," << trans_size << "," << " " << "\n";

        if (pe_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Finished !! Result Correct !!" << std::endl;
        }
    }

    outFile.close();
    status = aclrtDestroyStream(stream);
    return status;
}

int main(int argc, char *argv[])
{
    int status = 0;
    int n_pes = atoi(argv[INDEX1]);
    int pe_id = atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    g_npus = atoi(argv[INDEX4]);
    f_pe = atoi(argv[INDEX5]);
    f_npu = atoi(argv[INDEX6]);
    data_type = argv[INDEX7];
    perf_times = atoi(argv[INDEX8]);

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
    } else if (std::string(data_type) == "int32_t") {
        status = test_aclshmem_all_gather<int32_t>(pe_id, n_pes);
    } else if (std::string(data_type) == "float16_t") {
        status = test_aclshmem_all_gather<fp16_t>(pe_id, n_pes);
    } else if (std::string(data_type) == "bfloat16_t") {
        status = test_aclshmem_all_gather<bfloat16>(pe_id, n_pes);
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