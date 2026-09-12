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

#include "utils.h"
#include "param.h"

#include "acl/acl.h"
#include "shmem.h"
#include "multi_instance_kernel.h"

int g_npus = 8;
const char *ipport = "tcp://127.0.0.1:8998";
int f_pe = 0;
int f_npu = 0;
int device_id = 0;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr uint32_t DATA_SIZE_THRESHOLD = 2097152;
constexpr uint32_t BLOCK_NUM_SMALL_DATA = 8;
constexpr uint32_t BLOCK_NUM_LARGE_DATA = 16;
static char g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};

aclshmemx_uniqueid_t default_flag_uid;

template <class T>
int group_all_gather(uint64_t instance_id)
{
    // ACLStream init
    int status = 0;
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);

    // Prepare Kernel need
    uint64_t fftsAddr = util_get_ffts_config();

    int case_num = 20;
    std::vector<uint32_t> test_cases = {};
    for (int i = 0; i < case_num; i++) {
        int data_len = 16 * (1 << i);
        test_cases.push_back(data_len);
    }

    std::ofstream outFile("./result.csv");
    if (!outFile.is_open()) {
        std::cerr << "错误：无法创建文件！" << std::endl;
        return 1;
    }
    outFile << "M,N,Time(us)\n";

    int pe_id = aclshmem_my_pe();
    int n_pes = aclshmem_n_pes();

    // magic is used to sync.
    int magic = 1;

    for (int i = 0; i < test_cases.size(); i++) {
        if (aclshmem_my_pe() == 0) {
            printf("Instance %lu, Case: %d Started. \n", instance_id, test_cases[i]);
        }

        uint32_t trans_size = test_cases[i];

        uint32_t BLOCK_NUM = 8;
        //  Small data kernel needs 8 AIV core, Big data kernel needs 16 AIV.
        if (trans_size * sizeof(T) < DATA_SIZE_THRESHOLD) {
            BLOCK_NUM = BLOCK_NUM_SMALL_DATA;
        } else {
            BLOCK_NUM = BLOCK_NUM_LARGE_DATA;
        }

        // Host Data Prepare
        uint8_t *input_host;
        aclrtMallocHost(reinterpret_cast<void**>(&input_host), trans_size * sizeof(T));
        std::string inputFile = "../../examples/multi_instance/golden/allgather_" + std::to_string(trans_size)
            + "_" + std::to_string(n_pes) + "/input_gm_" + std::to_string(pe_id) + ".bin";
        ReadFile(inputFile, input_host, trans_size * sizeof(T));

        T *output_host;
        size_t output_size = n_pes * trans_size * sizeof(T);
        status = aclrtMallocHost(reinterpret_cast<void**>(&output_host), output_size);

        T *golden_host;
        status = aclrtMallocHost(reinterpret_cast<void**>(&golden_host), output_size);
        std::string goldenFile = "../../examples/multi_instance/golden/allgather_" + std::to_string(trans_size)
            + "_" + std::to_string(n_pes) + "/golden.bin";
        ReadFile(goldenFile, golden_host, n_pes * trans_size * sizeof(T));

        // Kernel Input Prepare
        void *input_ptr;
        aclrtMalloc(&input_ptr, trans_size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(input_ptr, trans_size * sizeof(T), input_host, trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

        void *output_ptr;
        aclrtMalloc(&output_ptr, trans_size * n_pes * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

        // sync Buffer + data Buffer
        int aiv_num = BLOCK_NUM;
        void *ptr = aclshmem_malloc(aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T));

        // AllGather
        int PERF_TIMES = 50;
        for (int zz = 0; zz < PERF_TIMES; zz++) {
            magic++;
            group_allgather<T>(BLOCK_NUM, stream, fftsAddr, 
                (uint8_t *)input_ptr, (uint8_t *)output_ptr, (uint8_t *)ptr, trans_size, magic * 1024, instance_id);
        }
        status = aclrtSynchronizeStream(stream);

        // Kernel Output
        status = aclrtMemcpy(output_host, output_size, output_ptr, output_size, ACL_MEMCPY_DEVICE_TO_HOST);

        // Result Check
        for (int zz = 0; zz < n_pes * trans_size; zz++) {
            if (static_cast<float>(output_host[zz]) != static_cast<float>(golden_host[zz])) {
                std::cout << static_cast<float>(output_host[zz]) << " != " << static_cast<float>(golden_host[zz])
                          << ", trans_size is : " << trans_size << ", idx is: " << zz
                          << ", pe_id is: "<< pe_id << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        aclshmem_free(ptr);
        aclrtFree(output_ptr);
        aclrtFree(input_ptr);

        // Host Ptr Free
        status = aclrtFreeHost(golden_host);
        status = aclrtFreeHost(output_host);
        status = aclrtFreeHost(input_host);

        outFile << 1 << "," << trans_size << "," << " " << "\n";

        if (aclshmem_my_pe() == 0) {
            printf("Instance %lu, Case: %d Finished !! Result Correct !!. \n", instance_id, test_cases[i]);
        }
    }

    outFile.close();

    status = aclrtDestroyStream(stream);
    return status;
}

static int aclshmem_instance_create_test(int pe_id, aclshmemx_init_attr_t &attr, std::vector<int> &dev_list)
{
    int status = 0;
    if (std::find(dev_list.begin(), dev_list.end(), pe_id) != dev_list.end()) {
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;
        ipport = "tcp://127.0.0.1:0"; // port must be 0 in default mode

        int local_pe_id = std::distance(dev_list.begin(), std::find(dev_list.begin(), dev_list.end(), pe_id));
        status = test_set_attr(local_pe_id, dev_list.size(), local_mem_size, ipport, default_flag_uid, &attr);

        // multi_instance default mode need comm_args is nullptr
        attr.comm_args = nullptr;
        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);

        void *ptr = aclshmem_malloc(1024);
        std::vector<uint64_t> copy_arr(1024 / sizeof(uint64_t), 1);
        status = aclrtMemcpy(ptr, 1024, copy_arr.data(), 1024, ACL_MEMCPY_HOST_TO_DEVICE);
        if (status != 0) {
            printf("aclshmem_malloc failed !! \n");
        }

        aclshmem_free(ptr);
        printf("Instance id : %ld malloc and free success !! \n", attr.instance_id);

        status = group_all_gather<int>(attr.instance_id);
    }
    return status;
}

static int aclshmem_instance_destroy_test(int dev_id, aclshmemx_init_attr_t &attr, std::vector<int> &dev_list)
{
    int status = 0;
    if (std::find(dev_list.begin(), dev_list.end(), dev_id) != dev_list.end()) {
        status = aclshmem_finalize(attr.instance_id);
    }
    return status;
}

int main(int argc, char *argv[])
{
    int status = 0;
    int n_pes = atoi(argv[INDEX1]);
    int pe_id = atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    f_npu  = atoi(argv[INDEX4]);

    // Acl && Shmem init
    device_id = pe_id % g_npus + f_npu;
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);

    // ### Instance 1 Init And Finalize
    aclshmemx_init_attr_t inst1_attr;
    inst1_attr.instance_id = 1;
    std::vector<int> inst1_devices = {0, 1, 2, 3};
    status = aclshmem_instance_create_test(pe_id, inst1_attr, inst1_devices);
    status = aclshmem_instance_destroy_test(pe_id, inst1_attr, inst1_devices);

    // ### Instance 2 Init
    aclshmemx_init_attr_t inst2_attr;
    inst2_attr.instance_id = 2;
    std::vector<int> inst2_devices = {0, 3};
    status = aclshmem_instance_create_test(pe_id, inst2_attr, inst2_devices);

    // ### Instance 3 Init And Finalize
    aclshmemx_init_attr_t inst3_attr;
    inst3_attr.instance_id = 3;
    std::vector<int> inst3_devices = {1, 3};
    status = aclshmem_instance_create_test(pe_id, inst3_attr, inst3_devices);
    status = aclshmem_instance_destroy_test(pe_id, inst3_attr, inst3_devices);
    
    // ### Instance 4 Init
    aclshmemx_init_attr_t inst4_attr;
    inst4_attr.instance_id = 4;
    std::vector<int> inst4_devices = {2, 3};
    status = aclshmem_instance_create_test(pe_id, inst4_attr, inst4_devices);

    // ### Instance 5 Init
    aclshmemx_init_attr_t inst5_attr;
    inst5_attr.instance_id = 5;
    std::vector<int> inst5_devices = {1, 2};
    status = aclshmem_instance_create_test(pe_id, inst5_attr, inst5_devices);

    // ### Instance 2,4,5 Finalize
    status = aclshmem_instance_destroy_test(pe_id, inst2_attr, inst2_devices);
    status = aclshmem_instance_destroy_test(pe_id, inst4_attr, inst4_devices);
    status = aclshmem_instance_destroy_test(pe_id, inst5_attr, inst5_devices);

    status = aclrtResetDevice(pe_id);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    return 0;
}