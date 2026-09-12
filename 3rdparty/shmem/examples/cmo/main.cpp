/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <sys/stat.h>

#include "utils.h"
#include "param.h"
#include "shmem.h"
#include "acl/acl.h"
#include "kernel_operator.h"

#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

#define CHECK_RET(func)                                                                                                \
    do {                                                                                                               \
        int ret = func;                                                                                                \
        if (ret != 0) {                                                                                                \
            std::cerr << __FILE__ << ":" << __LINE__ << " error: " << ret << std::endl;                                \
            return ret;                                                                                                \
        }                                                                                                              \
    } while (0)

enum class CMOEXAMPLE : uint32_t {
     NO_PREFETCH,
     HOST_PREFETCH,
     DEVICE_PREFETCH,
     DEVICE_BLOCK_PREFETCH,
 };

typedef struct {
    float avg_cmo_send_us;
    float avg_cmo_flag_us;
    float avg_copy_us;
    float total_band;
} res_t;

typedef struct {
    uint32_t loop_times;
    uint32_t copy_size_per_loop;
    uint32_t copypad_size;

    std::vector<float> no_prefetch_us;
    std::vector<float> host_prefetch_us;
    std::vector<float> device_block_prefetch_us;
    std::vector<float> device_block_prefetch_cmo_us;
    std::vector<float> device_block_prefetch_cmo_flag_us;
    std::vector<float> no_prefetch_bands;
    std::vector<float> host_prefetch_bands;
    std::vector<float> device_block_prefetch_bands;
} res_csv_t;

int g_npus = 8;
const char *ipport = "tcp://127.0.0.1:8998";
int f_pe = 0;
int f_npu = 0;
const char *data_type = "int";
static char g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};
aclshmemx_uniqueid_t default_flag_uid;
const uint32_t max_block_nums = 20;
const uint32_t gm_align_size = 512;
const uint32_t l2_cache_size = 192 * 1024 * 1024;

template <typename T>
__global__ __aicore__ void copy_perftest(GM_ADDR trash_gm, 
                                            GM_ADDR copy_gm, 
                                            GM_ADDR dump_gm, 
                                            GM_ADDR res_gm,
                                            uint32_t copypad_size, uint32_t copypad_times, uint32_t is_block_prefetch)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, dump_gm, ALL_DUMPSIZE);
#endif
    AscendC::TPipe pipe;
    uint32_t block_id = AscendC::GetBlockIdx();
    uint32_t n_blocks = AscendC::GetBlockNum();

    uint32_t copy_setp_size = copypad_size > gm_align_size ? copypad_size : gm_align_size;
    uint32_t copy_block_step_size = copy_setp_size * copypad_times;
    uint32_t copy_step_element = copy_setp_size / sizeof(T);
    uint32_t copy_block_step_element = copy_block_step_size / sizeof(T);
    uint32_t cmo_size = copy_block_step_size;

    uint64_t copy_ub;
    AscendC::LocalTensor<T> ub_copy_tensor;
    ub_copy_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_copy_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    AscendC::GlobalTensor<T> gm_tensor;
    gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(copy_gm));
    gm_tensor = gm_tensor[copy_block_step_element * block_id];

    // Define temporary UB buffer
    constexpr uint32_t ub_offset = 1024;
    constexpr uint32_t ub_size = 64;  // 64B for temporary buffer
    __ubuf__ uint8_t *tmp_buff = reinterpret_cast<__ubuf__ uint8_t *>(uint64_t(ub_offset));
    
    uint64_t start_cycle, end_cycle;
    uint64_t start_cmo_cycle, send_cmo_cycle, end_cmo_cycle;
    uint64_t start_copy_cycle, end_copy_cycle;

    start_cycle = AscendC::GetSystemCycle();
    start_cmo_cycle = start_cycle;
    
    if (is_block_prefetch != 0) {
        aclshmemx_cmo_nbi(reinterpret_cast<__gm__  uint8_t *>(copy_gm + cmo_size * block_id), cmo_size,
                            ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, EVENT_ID0);
    } else {
        aclshmemx_cmo_nbi(reinterpret_cast<__gm__  uint8_t *>(trash_gm + cmo_size * block_id), cmo_size,
                            ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, EVENT_ID0);
    }
    send_cmo_cycle = AscendC::GetSystemCycle();
    aclshmemx_sdma_quiet(tmp_buff, ub_size, EVENT_ID0);
    end_cmo_cycle = AscendC::GetSystemCycle();

    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::DataCopyExtParams data_copy_params(1, copypad_size, 0, 0, 0);
    AscendC::DataCopyPadExtParams<T> pad_params;

    start_copy_cycle = AscendC::GetSystemCycle();

    for(size_t i = 0; i < copypad_times; i++) {
        AscendC::DataCopyPad(ub_copy_tensor, gm_tensor[i * copy_step_element], data_copy_params, pad_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }

    end_copy_cycle = AscendC::GetSystemCycle();
    end_cycle = end_copy_cycle;

    AscendC::PipeBarrier<PIPE_ALL>();

    float cmo_send_us = ((float)(int32_t)(send_cmo_cycle - start_cmo_cycle) * 0.02f);
    float cmo_flag_us = ((float)(int32_t)(end_cmo_cycle - send_cmo_cycle) * 0.02f);
    float copy_us = ((float)(int32_t)(end_copy_cycle - start_copy_cycle) * 0.02f) / copypad_times;
    float copy_band = (float)(int32_t)(copypad_size) / copy_us / 1000.0f;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::PRINTF("[AIV=%d] GM->UB, copypad_size:%d Bytes, copypad_times:%d, cmo_send_us: %f us, cmo_flag_us: %f us, copy_us: %f us, band: %f GBps\n",
                    block_id, copypad_size, copypad_times, cmo_send_us, cmo_flag_us, copy_us, copy_band);
#endif

    AscendC::LocalTensor<float> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(tmp_buff);
    ub_tensor.address_.dataLen = ub_size;

    aclshmemi_set_value((__gm__ uint8_t *)(res_gm + (block_id * 10 + 0) * sizeof(float)), cmo_send_us, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
    aclshmemi_set_value((__gm__ uint8_t *)(res_gm + (block_id * 10 + 1) * sizeof(float)), cmo_flag_us, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
    aclshmemi_set_value((__gm__ uint8_t *)(res_gm + (block_id * 10 + 2) * sizeof(float)), copy_us, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
    aclshmemi_set_value((__gm__ uint8_t *)(res_gm + (block_id * 10 + 3) * sizeof(float)), copy_band, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <class T>
void copy_perftest_kernel(uint32_t block_dim, void* stream, 
                            uint8_t* trash_gm_ptr, 
                            uint8_t* cache_gm_ptr, 
                            uint8_t *dump_gm_ptr, 
                            uint8_t *res_gm_ptr,
                            uint32_t move_size_each_time, uint32_t copypad_times, uint32_t is_block_prefetch)
{
    copy_perftest<T><<<block_dim, nullptr, stream>>>(trash_gm_ptr,
                                                        cache_gm_ptr, 
                                                        dump_gm_ptr, 
                                                        res_gm_ptr, 
                                                        move_size_each_time, copypad_times, is_block_prefetch);
}

__global__ __aicore__ void cmo_pretech(GM_ADDR src, uint32_t size)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    // Define temporary UB buffer
    constexpr uint32_t ub_offset = 1024;
    constexpr uint32_t ub_size = 64;  // 64B for temporary buffer
    __ubuf__ uint8_t *tmp_buff = reinterpret_cast<__ubuf__ uint8_t *>(uint64_t(ub_offset));

    aclshmemx_cmo_nbi(reinterpret_cast<__gm__  uint8_t *>(src), size,
                    ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, EVENT_ID0);
    aclshmemx_sdma_quiet(tmp_buff, ub_size, EVENT_ID0);
}

void cmo_pretech_kernel(uint8_t* src, uint32_t size, void* stream)
{
    cmo_pretech<<<1, nullptr, stream>>>(src, size);
}

template <class T>
int copy_test(aclrtStream stream,
                uint32_t n_blocks,
                uint32_t copypad_size,
                uint32_t copypad_times,
                CMOEXAMPLE prefetch_type,
                res_t& res)
{
    uint32_t copy_setp_size = copypad_size > gm_align_size ? copypad_size : gm_align_size;
    uint32_t copy_block_size = copy_setp_size * copypad_times;
    size_t cache_gm_size = n_blocks * 2 * copy_block_size;

    void *cache_gm_ptr;
    CHECK_RET(aclrtMalloc((void **) &(cache_gm_ptr), cache_gm_size, ACL_MEM_MALLOC_HUGE_FIRST));

    size_t trash_gm_size = cache_gm_size > (size_t) l2_cache_size ? cache_gm_size : (size_t) l2_cache_size;
    void *trash_gm_ptr;
    CHECK_RET(aclrtMalloc((void **) &(trash_gm_ptr), trash_gm_size, ACL_MEM_MALLOC_HUGE_FIRST));

    void *device_dump;
#if defined(ENABLE_ASCENDC_DUMP)
    CHECK_RET(aclrtMalloc(reinterpret_cast<void **>(&device_dump), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
#endif
    void *res_ptr;
    size_t res_size = sizeof(float) * 2048;
    CHECK_RET(aclrtMalloc((void **) &(res_ptr), res_size, ACL_MEM_MALLOC_HUGE_FIRST));
    void *res_host;
    CHECK_RET(aclrtMallocHost((void **)(&res_host), res_size));

    uint32_t is_device_block_prefetch = 0;
    if (prefetch_type == CMOEXAMPLE::NO_PREFETCH) {
        is_device_block_prefetch = 0;
    } else if (prefetch_type == CMOEXAMPLE::DEVICE_PREFETCH) {
        cmo_pretech_kernel(reinterpret_cast<uint8_t *>(cache_gm_ptr), cache_gm_size, stream);
        CHECK_RET(aclrtSynchronizeStream(stream));
        aclshmem_barrier_all();
    } else if (prefetch_type == CMOEXAMPLE::DEVICE_BLOCK_PREFETCH) {
        is_device_block_prefetch = 1;
    } else if (prefetch_type == CMOEXAMPLE::HOST_PREFETCH){
        CHECK_RET(aclrtCmoAsync(cache_gm_ptr, cache_gm_size, ACL_RT_CMO_TYPE_PREFETCH, stream));
    }

    copy_perftest_kernel<T>(n_blocks, stream, 
        reinterpret_cast<uint8_t *>(trash_gm_ptr), 
        reinterpret_cast<uint8_t *>(cache_gm_ptr), 
        reinterpret_cast<uint8_t *>(device_dump), 
        reinterpret_cast<uint8_t *>(res_ptr), 
        copypad_size, copypad_times, is_device_block_prefetch);
    CHECK_RET(aclrtSynchronizeStream(stream));
    aclshmem_barrier_all();

    aclrtMemcpy(res_host, res_size, res_ptr, res_size, ACL_MEMCPY_DEVICE_TO_HOST);
    float* float_res = reinterpret_cast<float*>(res_host);
    float total_cmo_send_us = 0;
    float total_cmo_flag_us = 0;
    float total_copy_us = 0;
    float total_band = 0;
    for(int32_t i = 0; i < n_blocks * 2; i++)
    {
        total_cmo_send_us += float_res[i * 10 + 0];
        total_cmo_flag_us += float_res[i * 10 + 1];
        total_copy_us += float_res[i * 10 + 2];
        total_band += float_res[i * 10 + 3];
    }
    res.avg_cmo_send_us = total_cmo_send_us / (n_blocks * 2);
    res.avg_cmo_flag_us = total_cmo_flag_us / (n_blocks * 2);
    res.avg_copy_us = total_copy_us / (n_blocks * 2);
    res.total_band = total_band;
#if defined(ENABLE_ASCENDC_DUMP)
    Adx::AdumpPrintWorkSpace(device_dump, ALL_DUMPSIZE, stream, "cmo");
#endif
    CHECK_RET(aclrtFree(trash_gm_ptr));
    CHECK_RET(aclrtFree(cache_gm_ptr));
#if defined(ENABLE_ASCENDC_DUMP)
    CHECK_RET(aclrtFree(device_dump));
#endif
    CHECK_RET(aclrtFree(res_ptr));
    CHECK_RET(aclrtFreeHost(res_host));

    return 0;
}

static std::string format_size(int size) {
    if (size < 1024) {
        return std::to_string(size) + "B";
    } else if (size < 1024 * 1024) {
        size_t kb = size / 1024;
        std::ostringstream oss;
        oss << kb << "KB";
        return oss.str();
    } else {
        double mb = static_cast<double>(size) / (1024.0 * 1024.0);
        size_t mb_int = static_cast<size_t>(mb);
        std::ostringstream oss;
        oss << mb_int << "MB";
        return oss.str();
    }
}

template <class T>
int test_copy_perf(int my_pe, int n_pes)
{
    uint32_t copy_size_per_loop = 64 * 1024 * 1024; // less than L2 cache size
    uint32_t loop_times = 100;

    // ACLStream init
    aclrtStream stream = nullptr;
    CHECK_RET(aclrtCreateStream(&stream));

    std::vector<int> copypad_size_vector;   // 8B-256KB
    for (int exponent = 3; exponent <= 17; ++exponent) {
        int value = 1 << exponent;
        copypad_size_vector.push_back(value);
    }
    copypad_size_vector.push_back(192 * 1024);
    copypad_size_vector.push_back(256 * 1024);

    auto percentile = [](const std::vector<float>& v, float p) -> float {
        if (v.empty()) return 0.0f;
        std::vector<float> sorted = v;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = (size_t)(p / 100.0 * sorted.size());
        return sorted[idx];
    };

    std::vector<std::vector<std::string>> csv_data = {
        {"loop_times", "copy_size_per_loop", "blocks", "copypad_size", 
        "no_prefetch_time/us", "no_prefetch_band/Gbps",
        "host_prefetch_time/us", "host_prefetch_band/Gbps",
        "device_block_prefetch_time/us", "device_block_prefetch_band/Gbps"},
    };

    for (uint32_t n_blocks = 20; n_blocks <= max_block_nums; n_blocks++) {
        uint32_t aiv_num = n_blocks * 2;
        for (uint32_t copypad_size : copypad_size_vector) {
            
            uint32_t copypad_setp_size = copypad_size > gm_align_size ? copypad_size : gm_align_size;
            uint32_t copypad_times = copy_size_per_loop / aiv_num / copypad_setp_size;  
            uint32_t copy_block_size = copypad_setp_size * copypad_times;
            size_t cache_gm_size = aiv_num * copy_block_size;
            
            res_csv_t res_csv = {};
            std::vector<std::string> sub_data = {int_to_string(loop_times), format_size(copy_size_per_loop), int_to_string(n_blocks), format_size(copypad_size)};

            res_t res = {};
            // cmo warmup
            copy_test<T>(stream, n_blocks, copypad_size, copypad_times, CMOEXAMPLE::NO_PREFETCH, res);

            // no prefetch
            for (uint32_t loop_i = 0; loop_i < loop_times; loop_i++) {
                copy_test<T>(stream, n_blocks, copypad_size, copypad_times, CMOEXAMPLE::NO_PREFETCH, res);
                res_csv.no_prefetch_bands.push_back(res.total_band);
                res_csv.no_prefetch_us.push_back(res.avg_copy_us);
            }
            sub_data.push_back(float_to_string(percentile(res_csv.no_prefetch_us, 50.0f)));
            sub_data.push_back(float_to_string(percentile(res_csv.no_prefetch_bands, 50.0f)));

            // host prefetch
            for (uint32_t loop_i = 0; loop_i < loop_times; loop_i++) {
                copy_test<T>(stream, n_blocks, copypad_size, copypad_times, CMOEXAMPLE::HOST_PREFETCH, res);
                res_csv.host_prefetch_bands.push_back(res.total_band);
                res_csv.host_prefetch_us.push_back(res.avg_copy_us);
            }
            sub_data.push_back(float_to_string(percentile(res_csv.host_prefetch_us, 50.0f)));
            sub_data.push_back(float_to_string(percentile(res_csv.host_prefetch_bands, 50.0f)));

            // device block prefetch
            for (uint32_t loop_i = 0; loop_i < loop_times; loop_i++) {
                copy_test<T>(stream, n_blocks, copypad_size, copypad_times, CMOEXAMPLE::DEVICE_BLOCK_PREFETCH, res);
                res_csv.device_block_prefetch_bands.push_back(res.total_band);
                res_csv.device_block_prefetch_us.push_back(res.avg_copy_us);
            }
            sub_data.push_back(float_to_string(percentile(res_csv.device_block_prefetch_us, 50.0f)));
            sub_data.push_back(float_to_string(percentile(res_csv.device_block_prefetch_bands, 50.0f)));

            csv_data.push_back(sub_data);
        }
    }
    make_dir("output");
    write_csv("output/"+int_to_string(my_pe)+"_band.csv", csv_data);

    std::vector<std::vector<std::string>> csv_data_2 = {
        {"loop_times", "blocks", "cmo_size", 
        "cmo_send_time_p05/us", "cmo_send_time_p50/us", "cmo_send_time_p95/us",
        "cmo_flag_time_p05/us", "cmo_flag_time_p50/us", "cmo_flag_time_p95/us"},
    };

    std::vector<int> cmo_size_vector;   // 512B-96MB
    for (int exponent = 9; exponent <= 26; ++exponent) {
        int value = 1 << exponent;
        cmo_size_vector.push_back(value);
    }
    cmo_size_vector.push_back(96 * 1024 * 1024);

    for (uint32_t n_blocks = 1; n_blocks <= 1; n_blocks++) {
        uint32_t aiv_num = n_blocks * 2;
        uint32_t copypad_size = 512;
        for (uint32_t cmo_size : cmo_size_vector) {
            uint32_t copypad_times = cmo_size / copypad_size;
    
            res_csv_t res_csv = {};
            std::vector<std::string> sub_data = {int_to_string(loop_times), int_to_string(n_blocks), format_size(cmo_size)};

            res_t res = {};
            // cmo warmup
            copy_test<T>(stream, n_blocks, copypad_size, copypad_times, CMOEXAMPLE::NO_PREFETCH, res);

            // device block prefetch cmo
            for (uint32_t loop_i = 0; loop_i < loop_times; loop_i++) {
                copy_test<T>(stream, n_blocks, copypad_size, copypad_times, CMOEXAMPLE::DEVICE_BLOCK_PREFETCH, res);
                res_csv.device_block_prefetch_cmo_us.push_back(res.avg_cmo_send_us);
                res_csv.device_block_prefetch_cmo_flag_us.push_back(res.avg_cmo_flag_us);
            }
            sub_data.push_back(float_to_string(percentile(res_csv.device_block_prefetch_cmo_us, 5.0f)));
            sub_data.push_back(float_to_string(percentile(res_csv.device_block_prefetch_cmo_us, 50.0f)));
            sub_data.push_back(float_to_string(percentile(res_csv.device_block_prefetch_cmo_us, 95.0f)));
            sub_data.push_back(float_to_string(percentile(res_csv.device_block_prefetch_cmo_flag_us, 5.0f)));
            sub_data.push_back(float_to_string(percentile(res_csv.device_block_prefetch_cmo_flag_us, 50.0f)));
            sub_data.push_back(float_to_string(percentile(res_csv.device_block_prefetch_cmo_flag_us, 95.0f)));

            csv_data_2.push_back(sub_data);
        }
    }

    write_csv("output/"+int_to_string(my_pe)+"_cmo.csv", csv_data_2);
    std::cout << "PE " << my_pe << " Finished !" << std::endl;

    CHECK_RET(aclrtDestroyStream(stream));
    return 0;
}

int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                      aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(g_ipport) - 1);

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
    attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);
    aclshmemx_uniqueid_t *uid_args = (aclshmemx_uniqueid_t *)(attributes->comm_args);
    uid_args->my_pe = my_pe;
    uid_args->n_pes = n_pes;
    return ACLSHMEM_SUCCESS;
}

int main(int argc, char *argv[])
{
    int status = 0;
    int n_pes = atoi(argv[INDEX1]);
    int my_pe = atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    g_npus = atoi(argv[INDEX4]);
    f_pe = atoi(argv[INDEX5]);
    f_npu = atoi(argv[INDEX6]);
    data_type = argv[INDEX7];

    // Acl && Shmem init
    int32_t device_id = my_pe % g_npus + f_npu;
    CHECK_RET(aclInit(nullptr));
    CHECK_RET(aclrtSetDevice(device_id));

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    CHECK_RET(test_set_attr(my_pe, n_pes, local_mem_size, ipport, &attributes));

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;
    CHECK_RET(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    if (std::string(data_type) == "int") {
        CHECK_RET(test_copy_perf<int>(my_pe, n_pes));
    } else if (std::string(data_type) == "uint8") {
        CHECK_RET(test_copy_perf<uint8_t>(my_pe, n_pes));
    } else if (std::string(data_type) == "int64") {
        CHECK_RET(test_copy_perf<int64_t>(my_pe, n_pes));
    } else if (std::string(data_type) == "fp16") {
        CHECK_RET(test_copy_perf<half>(my_pe, n_pes));
    } else if (std::string(data_type) == "fp32") {
        CHECK_RET(test_copy_perf<float>(my_pe, n_pes));
    } else {
        printf("ERROR: Unsupport type\n");
        return -1;
    }

    CHECK_RET(aclshmem_finalize());
    CHECK_RET(aclrtResetDevice(device_id));
    CHECK_RET(aclFinalize());

    std::cout << "[SUCCESS] demo run success in pe " << my_pe << std::endl;
    return 0;
}