/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <csignal>
#include <iostream>
#include <algorithm>
#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"

int test_global_ranks;
int test_gnpu_num;
char test_global_ipport[ACLSHMEM_MAX_IP_PORT_LEN];
int test_first_rank;
int test_first_npu;
static char g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};
aclshmemx_uniqueid_t default_flag_uid;
int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                       aclshmemx_init_attr_t *attributes)
{
    SHM_ASSERT_RETURN(local_mem_size <= ACLSHMEM_MAX_LOCAL_SIZE, ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(n_pes <= ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(my_pe < ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(g_ipport) - 1);

        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            SHM_LOG_ERROR("my_pe:" << my_pe << " ip_port is nullptr!");
            return ACLSHMEM_INVALID_VALUE;
        }
    } else {
        SHM_LOG_WARN("init with my_pe:" << my_pe << " ip_port is nullptr!");
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);
    aclshmemx_uniqueid_t *uid_args = (aclshmemx_uniqueid_t *)(attributes->comm_args);

    return ACLSHMEM_SUCCESS;
}

void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st)
{
    *st = nullptr;
    int status = 0;
    if (n_ranks != (n_ranks & (~(static_cast<unsigned int>(n_ranks) - 1)))) {
        std::cout << "[TEST] input rank_size: "<< n_ranks << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);
    EXPECT_EQ(status = aclshmemx_set_conf_store_tls(false, nullptr, 0), 0);

    aclshmemx_init_attr_t attributes;
    
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);


    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, 0);
    *st = stream;
}

void test_cross_init(int pe_id, int n_pes, uint64_t local_mem_size, aclrtStream *st)
{
    *st = nullptr;
    int status = 0;
    if (n_pes != (n_pes & (~(static_cast<unsigned int>(n_pes) - 1)))) {
        std::cout << "[TEST] input pe_size: "<< n_pes << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = pe_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);
    EXPECT_EQ(status = aclshmemx_set_conf_store_tls(false, nullptr, 0), 0);

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, test_global_ipport, &attributes);
    attributes.option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_ROCE);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, 0);
    *st = stream;
}

int32_t test_rdma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st)
{
    *st = nullptr;
    int status = 0;
    if (n_ranks != (n_ranks & (~(static_cast<unsigned int>(n_ranks) - 1)))) {
        std::cout << "[TEST] input rank_size: "<< n_ranks << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);
    EXPECT_EQ(status = aclshmemx_set_conf_store_tls(false, nullptr, 0), 0);
    aclshmemx_init_attr_t attributes;
    
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);


    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, 0);
    *st = stream;
    return status;
}

int32_t test_sdma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st)
{
    *st = nullptr;
    int status = 0;
    if (n_ranks != (n_ranks & (~(static_cast<unsigned int>(n_ranks) - 1)))) {
        std::cout << "[TEST] input rank_size: "<< n_ranks << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);
    EXPECT_EQ(status = aclshmemx_set_conf_store_tls(false, nullptr, 0), 0);
    aclshmemx_init_attr_t attributes;

    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, 0);
    *st = stream;
    return status;
}

int32_t test_udma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st)
{
    *st = nullptr;
    int status = 0;
    if (n_ranks != (n_ranks & (~(static_cast<unsigned int>(n_ranks) - 1)))) {
        std::cout << "[TEST] input rank_size: " << n_ranks << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);
    EXPECT_EQ(status = aclshmemx_set_conf_store_tls(false, nullptr, 0), 0);
    aclshmemx_init_attr_t attributes;

    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, 0);
    *st = stream;
    return status;
}

void test_finalize(aclrtStream stream, int device_id)
{
    int status = aclshmem_finalize();
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclrtDestroyStream(stream), 0);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count)
{
    pid_t pids[process_count];
    int status[process_count];
    for (int i = 0; i < process_count; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            std::cout << "fork failed ! " << pids[i] << std::endl;
        } else if (pids[i] == 0) {
            func(i + test_first_rank, test_global_ranks, local_mem_size);
            raise(::testing::Test::HasFailure() ? SIGUSR2 : SIGUSR1);
        }
    }
    for (int i = 0; i < process_count; ++i) {
        waitpid(pids[i], &status[i], 0);
        if (WIFSIGNALED(status[i])) {
            int sig = WTERMSIG(status[i]);
            if (sig != SIGUSR1) {
                FAIL();
            }
        } else {
            FAIL();
        }
    }
}

int main(int argc, char** argv)
{
    int idx = 1;
    test_global_ranks = std::atoi(argv[idx++]);
    std::copy_n(argv[idx++], ACLSHMEM_MAX_IP_PORT_LEN - 1, test_global_ipport);
    test_global_ipport[ACLSHMEM_MAX_IP_PORT_LEN - 1] = '\0';
    test_gnpu_num = std::atoi(argv[idx++]);
    test_first_rank = std::atoi(argv[idx++]);
    test_first_npu = std::atoi(argv[idx++]);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}