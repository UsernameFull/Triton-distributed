/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>  
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iomanip>
#include <getopt.h>
#include <iomanip>
#include "utils.h"
#include "perftest_common_types.h"
#include "mte_perftest_common.h"

int g_npus = 8;
const char *ipport;
static const char *fuc_data_type;
static const char *fuc_test_type;
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;

static aclshmem_prof_pe_t *out_profs;
extern "C" void launch_mte_perf_kernel(uint32_t block_dim, void *stream, uint8_t *dst_gva, uint8_t *src_gva, int elements, int32_t frame_id, int test_mode, int data_type, int ub_size_kb, int64_t prof_pe_val, int loop_count);

static perftest::mte_mode_t get_mte_mode(const char *test_type_str) {
    if (strcmp(test_type_str, "put") == 0) return perftest::TEST_MODE_MTE_PUT;
    else if (strcmp(test_type_str, "bi_put") == 0) return perftest::TEST_MODE_BI_PUT;
    else if (strcmp(test_type_str, "get") == 0) return perftest::TEST_MODE_MTE_GET;
    else if (strcmp(test_type_str, "bi_get") == 0) return perftest::TEST_MODE_BI_GET;
    return perftest::TEST_MODE_MTE_PUT;
}

static aclshmem_mem_type_t get_memory_type(const char *memory_type_str) {
    if (strcmp(memory_type_str, "dram") == 0) return HOST_SIDE;
    return DEVICE_SIDE;
}

template<typename T>
int test_shmem_mte_perf_test_impl(int pe_id, int n_pes, uint64_t local_mem_size, 
                                   int min_block_size, int max_block_size,
                                   int min_exponent, int max_exponent, int loop_count, perftest::mte_mode_t test_mode, perftest::perf_data_type_t data_type_enum, int prof_pe, int ub_size_kb, aclshmem_mem_type_t memory_type, std::vector<std::vector<std::string>>& csv_data)
{
    int32_t device_id = (pe_id % g_npus + f_npu);
    int status = 0;
    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    int frame_id = 0;
    for (int block_size = min_block_size; block_size <= max_block_size; block_size++) {
        for (int exponent = min_exponent; exponent <= max_exponent; exponent++) {
            int datasize = std::pow(2, exponent);
            std::cout << "pe: " << pe_id << " block_size: " << block_size << " size: " << datasize << " frame_id: " << frame_id << std::endl;
            
            void *dst_ptr = aclshmemx_malloc(datasize * block_size, memory_type);
            void *src_ptr = aclshmemx_malloc(datasize * block_size, memory_type);

            int all_size = datasize * block_size;
            int trans_size = all_size / sizeof(T);

            std::vector<T> src_input(trans_size, 0);
            std::vector<T> dst_input(trans_size, 0);
            for (int i = 0; i < trans_size; i++) {
                src_input[i] = (T)(pe_id + 10);
                dst_input[i] = (T)(pe_id + 100);
            }

            status = aclrtMemcpy(src_ptr, all_size,
                                src_input.data(), all_size, ACL_MEMCPY_HOST_TO_DEVICE);
            status = aclrtMemcpy(dst_ptr, all_size,
                                dst_input.data(), all_size, ACL_MEMCPY_HOST_TO_DEVICE);

            launch_mte_perf_kernel(block_size, stream, (uint8_t *)dst_ptr, (uint8_t *)src_ptr, trans_size, frame_id, static_cast<int>(test_mode), static_cast<int>(data_type_enum), ub_size_kb, prof_pe, loop_count);
            status = aclrtSynchronizeStream(stream);

            bool verify_success = true;
            
            auto compare_values = [&](T *ptr1, T *ptr2, size_t count, const char *label1, const char *label2) -> bool {
                for (size_t i = 0; i < count; i++) {
                    if (ptr1[i] != ptr2[i]) {
                        std::cout << "  [ERROR] Mismatch at index " << i << ": " << label1 << "=" << (double)ptr1[i] << ", " << label2 << "=" << (double)ptr2[i] << std::endl;
                        return false;
                    }
                }
                return true;
            };

            std::vector<T> dst_host(trans_size, 0);
            std::vector<T> src_host(trans_size, 0);

            status = aclrtMemcpy(dst_host.data(), all_size, dst_ptr, all_size, ACL_MEMCPY_DEVICE_TO_HOST);
            status = aclrtMemcpy(src_host.data(), all_size, src_ptr, all_size, ACL_MEMCPY_DEVICE_TO_HOST);

            int peer_pe = (pe_id + 1) % n_pes;
            bool is_unilateral = (test_mode == perftest::TEST_MODE_MTE_PUT || test_mode == perftest::TEST_MODE_MTE_GET);
            bool is_put = (test_mode == perftest::TEST_MODE_MTE_PUT || test_mode == perftest::TEST_MODE_BI_PUT);

            if (test_mode == perftest::TEST_MODE_MTE_PUT) {
                std::cout << "\n[Verification] put operation: Checking data transfer..." << std::endl;
                if (pe_id != prof_pe) {
                    T expected_val = static_cast<T>(prof_pe + 10);
                    if (!compare_values(dst_host.data(), &expected_val, 1, "dst[0]", "peer_src[0]")) {
                        verify_success = false;
                        std::cout << "  [ERROR] put operation: destination data does not match source data!" << std::endl;
                    }
                }
            } else if (test_mode == perftest::TEST_MODE_MTE_GET) {
                std::cout << "\n[Verification] get operation: Checking data transfer..." << std::endl;
                if (pe_id == prof_pe) {
                    T expected_val = static_cast<T>(peer_pe + 100);
                    if (!compare_values(src_host.data(), &expected_val, 1, "src[0]", "peer_dst[0]")) {
                        verify_success = false;
                        std::cout << "  [ERROR] get operation: source data does not match destination data!" << std::endl;
                    }
                }
            } else if (test_mode == perftest::TEST_MODE_BI_PUT) {
                std::cout << "\n[Verification] bi_put operation: Checking data transfer..." << std::endl;
                T expected_val = static_cast<T>(peer_pe + 10);
                if (!compare_values(dst_host.data(), &expected_val, 1, "dst[0]", "peer_src[0]")) {
                    verify_success = false;
                    std::cout << "  [ERROR] bi_put operation: destination data does not match source data!" << std::endl;
                }
            } else if (test_mode == perftest::TEST_MODE_BI_GET) {
                std::cout << "\n[Verification] bi_get operation: Checking data transfer..." << std::endl;
                T expected_val = static_cast<T>(peer_pe + 100);
                if (!compare_values(src_host.data(), &expected_val, 1, "src[0]", "peer_dst[0]")) {
                    verify_success = false;
                    std::cout << "  [ERROR] bi_get operation: source data does not match destination data!" << std::endl;
                }
            }

            if (verify_success) {
                std::cout << "[Verification] SUCCESS: Data transferred correctly!" << std::endl;
            } else {
                std::cout << "[Verification] FAILED: Data transfer verification failed!" << std::endl;
            }
            std::cout << "" << std::endl;

            aclshmemx_show_prof(&out_profs, false);
            collect_prof_data_to_csv(out_profs, frame_id, block_size, datasize, g_npus, ub_size_kb, csv_data);

            aclshmemx_free(dst_ptr, memory_type);
            aclshmemx_free(src_ptr, memory_type);
            status = aclrtSynchronizeStream(stream);
            
            frame_id++;
            if (frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
                std::cerr << "警告: frame_id 超过最大值 " << ACLSHMEM_CYCLE_PROF_FRAME_CNT << ", 停止测试" << std::endl;
                break;
            }
        }
        if (frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
            break;
        }
    }
    aclshmemx_show_prof(nullptr, true);

    status = aclshmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();

    return 0;
}

int main(int argc, char *argv[])
{
    int status = 0;
    int n_pes = 2;
    int pe_id = 0;
    ipport = "tcp://127.0.0.1:8764";
    g_npus = 2;
    f_pe = 0;
    f_npu = 4;
    const char *test_type = "put";
    fuc_data_type = "float";
    const char *memory_type = "hbm";
    int min_block_size = 32;
    int max_block_size = 32;
    int min_exponent = 3;
    int max_exponent = 17;
    int loop_count = 1000;
    int ub_size_kb = 16;
    
    static struct option long_options[] = {
        {"pes", required_argument, 0, 0},
        {"pe-id", required_argument, 0, 0},
        {"ipport", required_argument, 0, 0},
        {"gnpus", required_argument, 0, 0},
        {"fpe", required_argument, 0, 0},
        {"fnpu", required_argument, 0, 0},
        {"test-type", required_argument, 0, 't'},
        {"datatype", required_argument, 0, 'd'},
        {"block-size", required_argument, 0, 'b'},
        {"block-range", required_argument, 0, 0},
        {"exponent", required_argument, 0, 'e'},
        {"exponent-range", required_argument, 0, 0},
        {"loop-count", required_argument, 0, 0},
        {"ub-size", required_argument, 0, 0},
        {"memory-type", required_argument, 0, 0},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "t:d:b:e:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 't':
                test_type = optarg;
                break;
            case 'd':
                fuc_data_type = optarg;
                break;
            case 'b':
                min_block_size = max_block_size = std::atoi(optarg);
                break;
            case 'e':
                min_exponent = max_exponent = std::atoi(optarg);
                break;
            case 0:
                if (strcmp(long_options[option_index].name, "pes") == 0) {
                    n_pes = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "pe-id") == 0) {
                    pe_id = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "ipport") == 0) {
                    ipport = optarg;
                } else if (strcmp(long_options[option_index].name, "gnpus") == 0) {
                    g_npus = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "fpe") == 0) {
                    f_pe = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "fnpu") == 0) {
                    f_npu = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "block-range") == 0) {
                    min_block_size = std::atoi(optarg);
                    if (optind < argc) {
                        max_block_size = std::atoi(argv[optind]);
                        optind++;
                    }
                } else if (strcmp(long_options[option_index].name, "exponent-range") == 0) {
                    min_exponent = std::atoi(optarg);
                    if (optind < argc) {
                        max_exponent = std::atoi(argv[optind]);
                        optind++;
                    }
                } else if (strcmp(long_options[option_index].name, "loop-count") == 0) {
                    loop_count = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "ub-size") == 0) {
                    ub_size_kb = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "memory-type") == 0) {
                    memory_type = optarg;
                }
                break;
            default:
                std::cerr << "错误: 未知参数" << std::endl;
                std::cerr << "使用方法: " << argv[0] << " --pes <n_pes> --pe-id <pe_id> --ipport <ip:port> --gnpus <gnpu_num> --fpe <first_pe> --fnpu <first_npu> [-t <put|bi_put|get|bi_get>] [-d <float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char>] [-b <block_size>] [-e <exponent>] [--block-range <min> <max>] [--exponent-range <min> <max>] [--loop-count <count>] [--ub-size <size>] [--memory-type <hbm|dram>]" << std::endl;
                return 1;
        }
    }

    if (strcmp(memory_type, "hbm") != 0 && strcmp(memory_type, "dram") != 0) {
        std::cerr << "错误: memory-type 必须是 hbm 或 dram" << std::endl;
        return 1;
    }
    
    std::cout << "[SUCCESS] demo run start in pe " << pe_id << ", test type: " << test_type << ", data type: " << fuc_data_type << std::endl;
    std::cout << "n_pes: " << n_pes << ", pe_id: " << pe_id << ", g_npus: " << g_npus << std::endl;
    std::cout << "核数范围: " << min_block_size << "-" << max_block_size << std::endl;
    std::cout << "幂数范围: " << min_exponent << "-" << max_exponent << std::endl;
    std::cout << "循环次数: " << loop_count << std::endl;
    std::cout << "UB size (KB): " << ub_size_kb << std::endl;
    std::cout << "Memory type: " << memory_type << std::endl;
    
    fuc_test_type = test_type;
    perftest::mte_mode_t test_mode = get_mte_mode(test_type);
    perftest::perf_data_type_t data_type_enum = get_data_type(fuc_data_type);
    aclshmem_mem_type_t memory_type_enum = get_memory_type(memory_type);
    
    uint64_t max_datasize = (1ULL << max_exponent);
    uint64_t max_required_size = max_datasize * max_block_size * 2;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    const uint64_t ONE_GB = 1024UL * 1024UL * 1024;
    const uint64_t MAX_GB = 40;
    
    if (max_required_size > local_mem_size) {
        uint64_t gb_needed = (max_required_size + ONE_GB - 1) / ONE_GB;
        if (gb_needed > MAX_GB) {
            std::cerr << "Error: Required memory exceeds 40GB! Max need " << gb_needed << " GB" << std::endl;
            std::cerr << "Please adjust block-range or exponent-range parameters" << std::endl;
            return 1;
        }
        local_mem_size = gb_needed * ONE_GB;
        std::cout << "INFO: Auto-adjust local_mem_size to " << gb_needed << " GB" << std::endl;
    }
    
    const char *prof_pe_env = std::getenv("SHMEM_CYCLE_PROF_PE");
    int prof_pe = 0;
    if (prof_pe_env != nullptr) {
        prof_pe = std::atoi(prof_pe_env);
    }
    
    std::vector<std::vector<std::string>> csv_data = {
        {"DataSize/B", "Npus", "Blocks", "UBsize/KB", "Bandwidth/GB/s", "CoreMaxTime/us", "SingleCoreTime/us"},
    };
    
    #define TEST_IMPL_OP(type) \
        status = test_shmem_mte_perf_test_impl<type>(pe_id, n_pes, local_mem_size, \
                                                      min_block_size, max_block_size, \
                                                      min_exponent, max_exponent, loop_count, test_mode, data_type_enum, prof_pe, ub_size_kb, memory_type_enum, csv_data)
    
    DISPATCH_BY_TYPE(fuc_data_type, TEST_IMPL_OP);
    
    #undef TEST_IMPL_OP
    
    if (pe_id == prof_pe) {
        std::string csv_filename = "output/" + std::string(fuc_test_type) + "_" + std::string(fuc_data_type) + "_" + int_to_string(prof_pe) + ".csv";
        write_csv(csv_filename, csv_data);
    }
    
    std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    
    return 0;
}
