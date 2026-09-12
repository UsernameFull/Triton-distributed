/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <acl/acl.h>
#include "shmem.h"

#if defined(RUN_WITH_UNIQUEID) || defined(RUN_WITH_UNIQUEID_MULTI_INSTANCE) || defined(RUN_WITH_MPI) || defined(RUN_UNIQUEID_WITH_DEFAULT)
#include <mpi.h>
#endif

#ifdef RUN_WITH_UNIQUEID
int run_main(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr << "Usage: " << argv[0] << " <pe> <pe_size>" << std::endl;
        return 1;
    }
    MPI_Init(nullptr, nullptr);
    int g_npu = atoi(argv[1]);
    int pe;
    int pe_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &pe);
    MPI_Comm_size(MPI_COMM_WORLD, &pe_size);
    int status = ACLSHMEM_SUCCESS;
    
    aclInit(nullptr);
    int device_id = pe % g_npu;
    aclrtSetDevice(device_id);
    
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t uid = ACLSHMEM_UNIQUEID_INITIALIZER;
    
    int64_t local_mem_size = 1024 * 1024 * 1024;
    if (pe == 0) {
        status = aclshmemx_get_uniqueid(&uid);
    }

    MPI_Bcast(&uid, sizeof(aclshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
    status = aclshmemx_set_attr_uniqueid_args(pe, pe_size, 
                                                local_mem_size, 
                                                &uid, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attributes);

    if (status != ACLSHMEM_SUCCESS) {
        std::cout << "[ERROR] pe " << pe << ": demo run failed!" << std::endl;
        aclrtResetDevice(pe);
        aclFinalize();
        MPI_Finalize();
        return 1;
    }
    std::cout << "pe " << pe << ": shmem init SUCCESS" << std::endl;

    status = shmem_finalize();
    aclrtResetDevice(pe);
    aclFinalize();
    MPI_Finalize();
    if (status != ACLSHMEM_SUCCESS) {
        std::cout << "[ERROR] pe " << pe << ": demo run failed!" << std::endl;
        return 1;
    } else {
        std::cout << "[SUCCESS] pe " << pe << ": demo run success!" << std::endl;
        return 0;
    }
}
#endif

#ifdef RUN_UNIQUEID_WITH_DEFAULT
int run_main(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr << "Usage: " << argv[0] << " <pe> <pe_size>" << std::endl;
        return 1;
    }
    MPI_Init(nullptr, nullptr);
    int g_npu = atoi(argv[1]);
    int pe;
    int pe_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &pe);
    MPI_Comm_size(MPI_COMM_WORLD, &pe_size);
    int status = ACLSHMEM_SUCCESS;
    
    aclInit(nullptr);
    int device_id = pe % g_npu;
    aclrtSetDevice(device_id);
    
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t uid = ACLSHMEM_UNIQUEID_INITIALIZER;
    
    int64_t local_mem_size = 1024 * 1024 * 1024;
    if (pe == 0) {
        status = aclshmemx_get_uniqueid(&uid);
    }

    MPI_Bcast(&uid, sizeof(aclshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
    status = aclshmemx_set_attr_uniqueid_args(pe, pe_size, 
                                                local_mem_size, 
                                                &uid, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    if (status != ACLSHMEM_SUCCESS) {
        std::cout << "[ERROR] pe " << pe << ": demo run failed!" << std::endl;
        aclrtResetDevice(pe);
        aclFinalize();
        MPI_Finalize();
        return 1;
    }
    std::cout << "pe " << pe << ": shmem init SUCCESS" << std::endl;

    status = shmem_finalize();
    aclrtResetDevice(pe);
    aclFinalize();
    MPI_Finalize();
    if (status != ACLSHMEM_SUCCESS) {
        std::cout << "[ERROR] pe " << pe << ": demo run failed!" << std::endl;
        return 1;
    } else {
        std::cout << "[SUCCESS] pe " << pe << ": demo run success!" << std::endl;
        return 0;
    }
}
#endif

#ifdef RUN_WITH_UNIQUEID_MULTI_INSTANCE
int uid_multi_instance_create(int pe_id, std::vector<int> &dev_list, uint64_t instance_id) {
    int status = 0;

    // Create MPI SubGroup
    int color;
    if (std::find(dev_list.begin(), dev_list.end(), pe_id) != dev_list.end()) {
        color = 1;
    } else {
        color = MPI_UNDEFINED;  // Won't be in Any SubGroup
    }

    MPI_Comm split_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, pe_id, &split_comm);

    // Create Shmem Instance
    if (std::find(dev_list.begin(), dev_list.end(), pe_id) != dev_list.end() && split_comm != MPI_COMM_NULL) {
        int local_pe_id = std::distance(dev_list.begin(), std::find(dev_list.begin(), dev_list.end(), pe_id));
        int pe_size = dev_list.size();
        uint64_t local_mem_size = 1024UL * 1024UL * 1024;

        aclshmemx_init_attr_t attr;
        aclshmemx_uniqueid_t uid = ACLSHMEM_UNIQUEID_INITIALIZER;

        if (local_pe_id == 0) {
            status = aclshmemx_get_uniqueid(&uid);
        }

        MPI_Bcast(&uid, sizeof(aclshmemx_uniqueid_t), MPI_UINT8_T, 0, split_comm);
        attr.instance_id = instance_id;
        status = aclshmemx_set_attr_uniqueid_args(
            local_pe_id, pe_size, local_mem_size, &uid, &attr);

        status = aclshmemx_init_attr(
            ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);

        if (status != ACLSHMEM_SUCCESS) {
            std::cout << "[ERROR] pe " << pe_id << ": create instance " << instance_id << " failed!" << std::endl;
            return 1;
        }
        std::cout << "pe " << pe_id << ": shmem create instance "<< instance_id << " init SUCCESS" << std::endl;
    }

    // Destroy MPI SubGroup
    if (split_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&split_comm);
    } else {
        // Do nothing
    }
    
    return 0;
}

int uid_multi_instance_destroy(int pe_id, std::vector<int> &dev_list, uint64_t instance_id) {
    int status = 0;
    if (std::find(dev_list.begin(), dev_list.end(), pe_id) != dev_list.end()) {
        status = aclshmem_finalize(instance_id);
        if (status == ACLSHMEM_SUCCESS) {
            std::cout << "pe " << pe_id << ": shmem finalize instance "<< instance_id << " init SUCCESS" << std::endl;
        }
    }
    return status;
}

int run_main(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr << "Usage: " << argv[0] << " <pe> <pe_size>" << std::endl;
        return 1;
    }
    MPI_Init(nullptr, nullptr);
    int g_npu = atoi(argv[1]);
    int pe;
    int pe_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &pe);
    MPI_Comm_size(MPI_COMM_WORLD, &pe_size);
    int status = ACLSHMEM_SUCCESS;
    
    aclInit(nullptr);
    int device_id = pe % g_npu;
    aclrtSetDevice(device_id);

    uint64_t inst1_id = 1;
    std::vector<int> inst1_devices = {0, 1, 2, 3};
    status = uid_multi_instance_create(pe, inst1_devices, inst1_id);

    uint64_t inst2_id = 2;
    std::vector<int> inst2_devices = {0, 2};
    status = uid_multi_instance_create(pe, inst2_devices, inst2_id);
    status = uid_multi_instance_destroy(pe, inst2_devices, inst2_id);

    status = uid_multi_instance_destroy(pe, inst1_devices, inst1_id);

    aclrtResetDevice(device_id);
    aclFinalize();
    MPI_Finalize();
    if (status != ACLSHMEM_SUCCESS) {
        std::cout << "[ERROR] pe " << pe << ": demo run failed!" << std::endl;
        return 1;
    } else {
        std::cout << "[SUCCESS] pe " << pe << ": demo run success!" << std::endl;
        return 0;
    }
}

#endif

#ifdef RUN_WITH_MPI
int run_main(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr << "Usage: " << argv[0] << " <pe> <pe_size>" << std::endl;
        return 1;
    }
    MPI_Init(nullptr, nullptr);
    int g_npu = atoi(argv[1]);
    int pe;
    int pe_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &pe);
    MPI_Comm_size(MPI_COMM_WORLD, &pe_size);
    int status = ACLSHMEM_SUCCESS;
    
    aclInit(nullptr);
    int device_id = pe % g_npu;
    aclrtSetDevice(device_id);
    
    uint64_t local_mem_size = 1024 * 1024 * 1024;
    aclshmemx_init_attr_t attributes = {
        pe, pe_size, "", local_mem_size, {0, ACLSHMEM_DATA_OP_MTE, 120, 120, 120}};
    
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_MPI, &attributes);

    if (status != ACLSHMEM_SUCCESS) {
        std::cout << "[ERROR] pe " << pe << ": demo run failed!" << std::endl;
        aclrtResetDevice(pe);
        aclFinalize();
        MPI_Finalize();
        return 1;
    }
    std::cout << "pe " << pe << ": shmem init SUCCESS" << std::endl;

    status = shmem_finalize();
    aclrtResetDevice(pe);
    aclFinalize();
    MPI_Finalize();
    if (status != ACLSHMEM_SUCCESS) {
        std::cout << "[ERROR] pe " << pe << ": demo run failed!" << std::endl;
        return 1;
    } else {
        std::cout << "[SUCCESS] pe " << pe << ": demo run success!" << std::endl;
        return 0;
    }
}
#endif

#ifdef RUN_WITH_DEFAULT
int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                       aclshmemx_uniqueid_t *default_flag_uid, aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), (size_t)(ACLSHMEM_MAX_IP_PORT_LEN - 1));

        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(default_flag_uid);

    return ACLSHMEM_SUCCESS;
}

int run_main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <pe> <pe_size>" << std::endl;
        return 1;
    }
    
    int device_id = atoi(argv[1]);
    int pe_size = atoi(argv[2]);
    std::string ipport = argv[3];
    int g_npu = atoi(argv[4]);
    int f_pe = atoi(argv[5]);
    int pe = f_pe + device_id;
    int f_npu = atoi(argv[6]);
    std::cout << pe << pe_size << ipport << std::endl;
    aclshmemx_uniqueid_t default_flag_uid = ACLSHMEM_UNIQUEID_INITIALIZER;
    aclshmemx_init_attr_t attributes;

    int status = ACLSHMEM_SUCCESS;
    aclInit(nullptr);
    aclrtSetDevice(device_id);
    
    uint64_t local_mem_size = 1024 * 1024 * 1024;
    test_set_attr(pe, pe_size, local_mem_size, ipport.c_str(), &default_flag_uid, &attributes);
      
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    if (status != ACLSHMEM_SUCCESS) {
        std::cout << "[ERROR] pe " << pe << ": demo run failed!" << std::endl;
        aclrtResetDevice(pe);
        aclFinalize();
        return 1;
    }
    std::cout << "pe " << pe << ": shmem init SUCCESS" << std::endl;

    status = shmem_finalize();
    aclrtResetDevice(pe);
    aclFinalize();
    if (status != ACLSHMEM_SUCCESS) {
        std::cout << "[ERROR] pe " << pe << ": demo run failed!" << std::endl;
        return 1;
    } else {
        std::cout << "[SUCCESS] pe " << pe << ": demo run success!" << std::endl;
        return 0;
    }
}
#endif

int main(int main_argc, char* main_argv[]) {
    int status = ACLSHMEM_SUCCESS;
    #ifdef RUN_WITH_DEFAULT
        status = run_main(main_argc, main_argv);
    #elif defined(RUN_WITH_UNIQUEID) || defined(RUN_WITH_UNIQUEID_MULTI_INSTANCE) || defined(RUN_WITH_MPI)|| defined(RUN_UNIQUEID_WITH_DEFAULT)
        status = run_main(main_argc, main_argv);
    #else
        std::cerr << "Error: Please define one of RUN_WITH_UNIQUEID/RUN_WITH_UNIQUEID_MULTI_INSTANCE/RUN_WITH_MPI/RUN_WITH_DEFAULT/RUN_UNIQUEID_WITH_DEFAULT" << std::endl;
        return 1;
    #endif
    return status;
}