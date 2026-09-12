/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <acl/acl.h>
#include <netinet/in.h>
#include <errno.h>
#include <unistd.h>
#include <gtest/gtest.h>
#include "shmemi_host_common.h"
#include "host/shmem_host_def.h"
#include "unittest_main_test.h"

namespace shm {
constexpr uint32_t timeout = 50;
void logger_test_example(int level, const char* msg)
{
    // do print here
}
}

void test_aclshmem_init(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(g_state.mype, rank_id);
    EXPECT_EQ(g_state.npes, n_ranks);
    EXPECT_NE(g_state.heap_base, nullptr);
    EXPECT_NE(g_state.p2p_device_heap_base[rank_id], nullptr);
    EXPECT_EQ(g_state.heap_size, local_mem_size + ACLSHMEM_EXTRA_SIZE);
    EXPECT_NE(g_state.team_pools[0], nullptr);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_IS_INITIALIZED);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_aclshmem_init_invalid_rank_id(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int erank_id = -1;
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    aclshmemx_init_attr_t attributes;
    test_set_attr(erank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);


    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, ACLSHMEM_INVALID_VALUE);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_aclshmem_init_invalid_n_ranks(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int en_ranks = ACLSHMEM_MAX_PES + 1;
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_uniqueid_t uid;

    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, en_ranks, local_mem_size, test_global_ipport, &attributes);


    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    EXPECT_EQ(status, ACLSHMEM_INVALID_VALUE);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_aclshmem_init_rank_id_over_size(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id + n_ranks, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_INVALID_PARAM);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_aclshmem_init_zero_mem(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    // local_mem_size = 0
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_INVALID_VALUE);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_shmem_init_huge_pool(int pe_id, int n_pes, uint64_t local_mem_size)
{
    uint32_t device_id = pe_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    size_t hbm_free, hbm_total;
    EXPECT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &hbm_free, &hbm_total), 0);
    if (local_mem_size > hbm_total) {
        constexpr uint64_t hbm_ratio_percent = 60;
        local_mem_size = ALIGN_UP(hbm_total * hbm_ratio_percent / 100, 1024ULL * 1024 * 1024);
    }
    shmem_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, test_global_ipport, &attributes);

    status = shmem_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(g_state.mype, pe_id);
    EXPECT_EQ(g_state.npes, n_pes);
    EXPECT_NE(g_state.heap_base, nullptr);
    EXPECT_NE(g_state.p2p_device_heap_base[pe_id], nullptr);
    EXPECT_EQ(g_state.heap_size, local_mem_size + ACLSHMEM_EXTRA_SIZE);
    EXPECT_NE(g_state.team_pools[0], nullptr);
    status = shmem_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_IS_INITIALIZED);
    status = shmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_shmem_init(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    status = shmem_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(g_state.mype, rank_id);
    EXPECT_EQ(g_state.npes, n_ranks);
    EXPECT_NE(g_state.heap_base, nullptr);
    EXPECT_NE(g_state.p2p_device_heap_base[rank_id], nullptr);
    EXPECT_EQ(g_state.heap_size, local_mem_size + ACLSHMEM_EXTRA_SIZE);
    EXPECT_NE(g_state.team_pools[0], nullptr);
    status = shmem_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_IS_INITIALIZED);
    status = shmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}
void aclshmem_test_logger(int level, const char *msg) {
    if (level < 1) {
        return;
    }
    if (msg == NULL) {
        printf("extern log set :(null)\n");
        return;
    }
    const char* level_name = "unknown";
    switch (level) {
        case 0: level_name = "debug"; break;
        case 1: level_name = "info"; break;
        case 2: level_name = "warn"; break;
        case 3: level_name = "error"; break;
        case 4: level_name = "fatal"; break;
        default: break;
    }

    printf("extern log set [%s]: %s\n", level_name, msg);
}

void test_aclshmem_extern_logger(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    EXPECT_EQ(aclshmemx_set_extern_logger(aclshmem_test_logger), ACLSHMEM_SUCCESS);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(g_state.mype, rank_id);
    EXPECT_EQ(g_state.npes, n_ranks);
    EXPECT_NE(g_state.heap_base, nullptr);
    EXPECT_NE(g_state.p2p_device_heap_base[rank_id], nullptr);
    EXPECT_EQ(g_state.heap_size, local_mem_size + ACLSHMEM_EXTRA_SIZE);
    EXPECT_NE(g_state.team_pools[0], nullptr);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_IS_INITIALIZED);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

void test_aclshmem_init_invalid_ip(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, "tcp://123.45.67.89:2345", &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_INNER_ERROR);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void test_aclshmem_repeatedly_init(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_INNER_ERROR);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    sleep(2);
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(g_state.mype, rank_id);
    EXPECT_EQ(g_state.npes, n_ranks);
    EXPECT_NE(g_state.heap_base, nullptr);
    EXPECT_NE(g_state.p2p_device_heap_base[rank_id], nullptr);
    EXPECT_EQ(g_state.heap_size, local_mem_size + ACLSHMEM_EXTRA_SIZE);
    EXPECT_NE(g_state.team_pools[0], nullptr);
    status = aclshmemx_init_status();
    EXPECT_EQ(status, ACLSHMEM_STATUS_IS_INITIALIZED);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

std::vector<std::string> split_env(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void test_aclshmem_init_cant_access(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    const std::string env_name = "ASCEND_RT_VISIBLE_DEVICES";
    std::string original_value;
    bool is_original_empty = false;
    const char* original_env_value = std::getenv(env_name.c_str());
    std::string target_device;
    if (original_env_value) {
        original_value = original_env_value;
        std::vector<std::string> devices = split_env(original_value, ',');
        if (devices.size() >= n_ranks) {
            target_device = devices[rank_id];
        } else {
            return;
        }
    } else {
        target_device = std::to_string(rank_id);
        is_original_empty = true;
    }
    setenv(env_name.c_str(), target_device.c_str(), 1);

    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = ACLSHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(0), 0);
    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    EXPECT_EQ(status, ACLSHMEM_DL_FUNC_FAILED);
    status = aclshmem_finalize();
    EXPECT_EQ(status, ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(0), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (is_original_empty) {
        unsetenv(env_name.c_str());
    } else {
        setenv(env_name.c_str(), original_value.c_str(), 1);
    }
}

TEST(TestInitAPI, TestShmemInit)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitHugePool)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 38 * 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init_huge_pool, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidRankId)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init_invalid_rank_id, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidNRanks)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init_invalid_n_ranks, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorRankIdOversize)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init_rank_id_over_size, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorZeroMem)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 0;
    test_mutil_task(test_aclshmem_init_zero_mem, local_mem_size, process_count);
}

TEST(TestInitAPI, TestInfoGetVersion)
{
    int major = 0;
    int minor = 0;
    aclshmem_info_get_version(&major, &minor);
    EXPECT_EQ(major, ACLSHMEM_MAJOR_VERSION);
    EXPECT_EQ(minor, ACLSHMEM_MINOR_VERSION);
}

TEST(TestInitAPI, TestInfoGetVersionNull)
{
    int major = 0;
    aclshmem_info_get_version(&major, nullptr);
    EXPECT_EQ(major, 0);
}

TEST(TestInitAPI, TestInfoGetName)
{
    char name[256];
    aclshmem_info_get_name(name);
    EXPECT_TRUE(strlen(name) > 0);

    std::string expect = "ACLSHMEM v" + std::to_string(ACLSHMEM_VENDOR_MAJOR_VER) + "."
        + std::to_string(ACLSHMEM_VENDOR_MINOR_VER) + "." + std::to_string(ACLSHMEM_VENDOR_PATCH_VER).c_str();
    for (size_t i = 0; i < expect.length(); i++) {
        EXPECT_EQ(expect[i], name[i]);
    }
}

TEST(TestInitAPI, TestInfoGetNameNull)
{
    char *input = nullptr;
    aclshmem_info_get_name(input);
    EXPECT_EQ(input, nullptr);
}

TEST(TestInitAPI, TestShmemSetLogLevel)
{
    auto ret = aclshmemx_set_log_level(aclshmem_log::DEBUG_LEVEL);
    EXPECT_EQ(ret, 0);

    char* original_log_level = NULL;
    const char* env_val = getenv("SHMEM_LOG_LEVEL");
    if (env_val != NULL) {
        original_log_level = strdup(env_val);
    }

    setenv("SHMEM_LOG_LEVEL", "DEBUG", 1);
    EXPECT_EQ(aclshmemx_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "INFO", 1);
    EXPECT_EQ(aclshmemx_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "WARN", 1);
    EXPECT_EQ(aclshmemx_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "ERROR", 1);
    EXPECT_EQ(aclshmemx_set_log_level(-1), 0);

    setenv("SHMEM_LOG_LEVEL", "FATAL", 1);
    EXPECT_EQ(aclshmemx_set_log_level(-1), 0);

    unsetenv("SHMEM_LOG_LEVEL");
    if (original_log_level != NULL) {
        setenv("SHMEM_LOG_LEVEL", original_log_level, 1);
        free(original_log_level);
        original_log_level = NULL;
    }
}

TEST(TestInitAPI, TestShmemInitAlias)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemExternLog)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_extern_logger, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemRepeatInit)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_repeatedly_init, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemGetUniqueIdAndInit)
{
    const uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    const int process_count = 1;

    uint32_t device_id = 0;

    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(aclrtSetDevice(device_id), 0);
    ASSERT_EQ(setenv("SHMEM_UID_SESSION_ID", "127.0.0.1:8666", 1), 0);
    aclshmemx_uniqueid_t uid;
    int ret = aclshmemx_get_uniqueid(&uid);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);
    EXPECT_EQ(uid.version, ACLSHMEM_UNIQUEID_VERSION);

    aclshmemx_init_attr_t attr;
    ret = aclshmemx_set_attr_uniqueid_args(0, 1, (int64_t)local_mem_size, &uid, &attr);
    EXPECT_EQ(ret, ACLSHMEM_SUCCESS);

    EXPECT_EQ(attr.my_pe, 0);
    EXPECT_EQ(attr.n_pes, 1);
    EXPECT_EQ(attr.local_mem_size, 64UL * 1024UL * 1024UL);
    EXPECT_NE(attr.comm_args, nullptr);

    EXPECT_EQ(aclshmem_finalize(), ACLSHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

TEST(TestInitAPI, TestShmemInvalidIP)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init_invalid_ip, local_mem_size, 1);
}

TEST(TestInitAPI, TestShmemCantAccess)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_aclshmem_init_cant_access, local_mem_size, process_count);
}