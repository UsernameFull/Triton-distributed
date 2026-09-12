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
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "utils.h"

int g_npus = 8;
const char *ipport;
int f_pe = 0;
int f_npu = 0;
const char *test_type;

extern void rdma_highlevel_put_pingpong_latency_do(uint32_t block_dim, void* st, uint64_t cfg, uint8_t* gva, int len);
extern void rdma_postsend_cost_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int len);
extern void rdma_highlevel_put_bw_do(uint32_t block_dim, void* stream, uint64_t cfg, uint8_t* gva, int len);
extern void rdma_mte_put_bw_do(uint32_t block_dim, void* stream, uint64_t cfg, uint8_t* gva, int len, int64_t iter);

aclshmemx_uniqueid_t default_flag_uid;

int test_aclshmem_rdma_highlevel_put_pingpong_latency(int pe_id, int n_pes, uint64_t local_mem_size, int message_length)
{
    uint32_t iteration = 1;
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const double ration50 = 50.0;
    const int times32 = 32;
    const int iterRange = 10;
    const int size6M = 6 * 1024 * 1024;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint64_t fftsConfig = util_get_ffts_config();
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_pes;

    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = pe_id + iterRange;
    }
    aclrtMemcpy(gva + pe_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(gva + n_pes * message_length + times32 * (pe_id + 1), times32,
        xHost, times32, ACL_MEMCPY_HOST_TO_DEVICE);

    for (uint32_t i = 0; i < iteration; i++) {
        rdma_highlevel_put_pingpong_latency_do(1, stream, fftsConfig, gva, message_length);
    }
    aclrtSynchronizeStream(stream);
    if (pe_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_pes,
            sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA highlevel put pingpong latency test. Message length = " << message_length
            << " Byte; latency = " << xHost[0] / ration50 << " us." << std::endl;
    }

    aclrtFreeHost(xHost);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

int test_aclshmem_rdma_postsend_cost(int pe_id, int n_pes, uint64_t local_mem_size, int message_length)
{
    uint32_t iteration = 1;
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const double ration2500 = 50.0 * 500;
    const int iterRange = 10;
    const int size6M = 6 * 1024 * 1024;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint64_t fftsConfig = util_get_ffts_config();
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_pes;

    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = pe_id + iterRange;
    }
    aclrtMemcpy(gva + pe_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    for (uint32_t i = 0; i < iteration; i++) {
        rdma_postsend_cost_do(1, stream, fftsConfig, gva, message_length);
    }
    aclrtSynchronizeStream(stream);
    if (pe_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_pes,
            sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA postsend cost test. Message length = " << message_length
            << " Byte; postsend cost = " << xHost[0] / ration2500 << " us." << std::endl;
    }

    aclrtFreeHost(xHost);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

int test_aclshmem_rdma_highlevel_put_bw(int pe_id, int n_pes, uint64_t local_mem_size, int message_length)
{
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const double ration50 = 50.0;
    const int iterRange = 10;
    const int size6M = 6 * 1024 * 1024;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint64_t fftsConfig = util_get_ffts_config();
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size6M));

    int64_t *xHost;
    size_t totalSize = message_length * n_pes;

    aclrtMallocHost(reinterpret_cast<void **>(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = pe_id + iterRange;
    }
    aclrtMemcpy(gva + pe_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    rdma_highlevel_put_bw_do(1, stream, fftsConfig, gva, message_length);
    aclrtSynchronizeStream(stream);
    if (pe_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_pes, sizeof(int64_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA high level put bandwidth test. Message length = " << message_length
            << " Byte; time = " << xHost[0] / ration50 << " us." << std::endl;
    }

    aclrtFreeHost(xHost);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

int test_aclshmem_rdma_mte_put_bw(int pe_id, int n_pes, uint64_t local_mem_size, int message_length)
{
    int32_t device_id = pe_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    const int size32M = 32 * 1024 * 1024;
    const int size128K = 128 * 1024;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // aclshmemx_mte_set_ub_params(0, 128 * 1024, 0);

    uint64_t fftsConfig = util_get_ffts_config();
    uint8_t *gva = static_cast<uint8_t*>(aclshmem_malloc(size32M));
    int64_t *inHost;
    int64_t *outHost;
    size_t totalSize = message_length * n_pes * 3;
    aclrtMallocHost(reinterpret_cast<void **>(&inHost), totalSize);
    aclrtMallocHost(reinterpret_cast<void **>(&outHost), totalSize);
    bzero(inHost, totalSize);
    double rdmaTotalTime = 0.0;
    double mteTotalTime = 0.0;
    const int mteIdx = 6;
    const double ratio10 = 10.0;
    const double ration50 = 50.0;
    const int dstMax = 64;
    const int iterRange = 10;
    const int maxIter = 20;
    const int peTimes = 2;

    for (int iter = 0; iter < maxIter; iter++) {
        for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
            inHost[i + pe_id * message_length / sizeof(int64_t)] = pe_id + iterRange + iter;
        }
        for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
            inHost[i + (pe_id + n_pes) * message_length / sizeof(int64_t)] = pe_id + iterRange + iter;
        }
        aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclshmemi_control_barrier_all();
        rdma_mte_put_bw_do(1, stream, fftsConfig, gva, message_length, iter);
        aclrtSynchronizeStream(stream);
        if (pe_id == 0 && iter >= iterRange) {
            aclrtMemcpy(outHost, dstMax, gva + message_length * n_pes * peTimes, dstMax, ACL_MEMCPY_DEVICE_TO_HOST);
            rdmaTotalTime += outHost[0] / ration50;
            mteTotalTime += outHost[mteIdx] / ration50;
        }
    }
    if (pe_id == 0) {
        std::cout << "RDMA rdma mte test. Message length = " << message_length << " Byte; average RDMA time = "
            << rdmaTotalTime / ratio10 << " us." << std::endl;
        std::cout << "RDMA rdma mte test. Message length = " << message_length << " Byte; average MTE time = "
            << mteTotalTime / ratio10 << " us." << std::endl;
    }

    aclrtFreeHost(inHost);
    aclrtFreeHost(outHost);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{
    const int expected_argc = 9;
    if (argc != expected_argc) {
        std::cout << "[ERROR] Paramater number mismatch." << std::endl;
        std::cout << "[USAGE] ./rdma_perftest <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> "
            << "<test_type> <msg_len>. See README for more details." << std::endl;
    }
    int sub = 1;
    int status = 0;
    int n_pes = atoi(argv[sub++]);
    const int pe_max = 2;
    if (n_pes != pe_max) {
        std::cout << "[ERROR] Error number of pes! Only support 2 pes!" << std::endl;
    }
    int pe_id = atoi(argv[sub++]);
    if (pe_id >= pe_max) {
        std::cout << "[ERROR] Error pe ID! Only support 2 pes!" << std::endl;
    }
    ipport = argv[sub++];
    g_npus = atoi(argv[sub++]);
    f_pe = atoi(argv[sub++]);
    f_npu = atoi(argv[sub++]);
    test_type = argv[sub++];
    int msg_len = atoi(argv[sub++]);
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    if (std::string(test_type) == "highlevel_put_pingpong_latency") {
        test_aclshmem_rdma_highlevel_put_pingpong_latency(pe_id, n_pes, local_mem_size, msg_len);
    } else if (std::string(test_type) == "postsend_cost") {
        test_aclshmem_rdma_postsend_cost(pe_id, n_pes, local_mem_size, msg_len);
    } else if (std::string(test_type) == "highlevel_put_bw") {
        test_aclshmem_rdma_highlevel_put_bw(pe_id, n_pes, local_mem_size, msg_len);
    } else if (std::string(test_type) == "rdma_mte_bw") {
        test_aclshmem_rdma_mte_put_bw(pe_id, n_pes, local_mem_size, msg_len);
    }

    std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    return 0;
}