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
#include <cstring>
#include <cerrno>
#include <algorithm>
#include <random>
#include <cmath>
#include <type_traits>

#include "acl/acl.h"
#include "kernel_operator.h"

#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

#include "utils.h"
#include "param.h"
#include "shmem.h"

#define CHECK_RET(func)                                                                                                \
    do {                                                                                                               \
        int ret = func;                                                                                                \
        if (ret != 0) {                                                                                                \
            std::cerr << __FILE__ << ":" << __LINE__ << " error: " << ret << std::endl;                                \
            return ret;                                                                                                \
        }                                                                                                              \
    } while (0)


int g_npus = 8;
const char *ipport = "tcp://127.0.0.1:8998";
int f_pe = 0;
int f_npu = 0;
const char *data_type = "int";
static char g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};
aclshmemx_uniqueid_t default_flag_uid;

constexpr float FLOAT_EPS = 1e-5f;
constexpr double DOUBLE_EPS = 1e-8;
constexpr int INT_EPS = 0;

template <typename T>
bool check_accuracy(T actual, T expected)
{
    if constexpr (std::is_integral_v<T>) {
        return actual == expected;
    } else if constexpr (std::is_same_v<T, half>) {
        return std::fabs(actual - expected) < FLOAT_EPS;
    } else if constexpr (std::is_same_v<T, double>) {
        return std::fabs(actual - expected) < DOUBLE_EPS;
    } else {
        return actual == expected;
    }
}

template <typename T>
__global__ __aicore__ void allgather_sdma(GM_ADDR gva, int elem_size, GM_ADDR dump, bool is_put)
{
    AscendC::TPipe pipe;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#endif
    if ASCEND_IS_AIV {
        int my_pe = aclshmem_my_pe();
        int n_pes = aclshmem_n_pes();

        // Define temporary UB buffer for SDMA operations
        constexpr uint32_t ub_offset = 1024;
        constexpr uint32_t ub_size = 64;  // 64B for temporary buffer
        __ubuf__ uint8_t *tmp_buff = reinterpret_cast<__ubuf__ uint8_t *>(uint64_t(ub_offset));

        uint32_t data_length = elem_size * sizeof(T);
        // allgather
        const auto cur_block_idx = AscendC::GetBlockIdx();
        const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint64_t base_per_core = data_length / comm_block_dim;
        uint64_t extra_bytes = data_length % comm_block_dim;
        uint64_t data_offset = 0;
        if (cur_block_idx < extra_bytes) {
            data_offset = cur_block_idx * (base_per_core + 1);
        } else {
            data_offset = extra_bytes * (base_per_core + 1) +
                                    (cur_block_idx - extra_bytes) * base_per_core;
        }
        if (cur_block_idx < extra_bytes) {
            base_per_core += 1;
        }
        if (base_per_core == 0) {
            return;
        }
        for (int i = 0; i < n_pes; i++) {
            if (i == my_pe) {
                continue;
            }
            if (is_put) {
                aclshmemx_sdma_put_nbi(gva + data_length * my_pe + data_offset, gva + data_length * my_pe + data_offset,
                    tmp_buff, ub_size, base_per_core, i, EVENT_ID0);
            } else {
                aclshmemx_sdma_get_nbi(gva + data_length * i + data_offset, gva + data_length * i + data_offset,
                    tmp_buff, ub_size, base_per_core, i, EVENT_ID0);
            }
        }
        aclshmemx_sdma_quiet(tmp_buff, ub_size, EVENT_ID0);
    }
}

template <typename T>
__global__ __aicore__ void allgather_sdma_tensor(GM_ADDR gva, int elem_size, GM_ADDR dump, bool is_put)
{
    AscendC::TPipe pipe;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#endif
    if ASCEND_IS_AIV {
        int my_pe = aclshmem_my_pe();
        int n_pes = aclshmem_n_pes();

        // Define temporary UB buffer as LocalTensor for SDMA operations
        constexpr uint32_t ub_offset = 1024;
        constexpr uint32_t ub_size = 64;  // 64B for temporary buffer
        AscendC::LocalTensor<T> tmp_local;
        tmp_local.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
        tmp_local.address_.bufferAddr = ub_offset;
        tmp_local.address_.dataLen = ub_size;

        const auto cur_block_idx = AscendC::GetBlockIdx();
        const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint64_t base_per_core = elem_size / comm_block_dim;
        uint64_t extra_size = elem_size % comm_block_dim;
        uint64_t data_offset = 0;
        if (cur_block_idx < extra_size) {
            data_offset = cur_block_idx * (base_per_core + 1);
        } else {
            data_offset = extra_size * (base_per_core + 1) +
                                    (cur_block_idx - extra_size) * base_per_core;
        }
        if (cur_block_idx < extra_size) {
            base_per_core += 1;
        }
        if (base_per_core == 0) {
            return;
        }
        for (int i = 0; i < n_pes; i++) {
            if (i == my_pe) {
                continue;
            }
            AscendC::GlobalTensor<T> src_tensor;
            AscendC::GlobalTensor<T> dst_tensor;

            if (is_put) {
                __gm__ T* data_addr =
                    reinterpret_cast<__gm__ T*>(gva + my_pe * elem_size * sizeof(T) + data_offset * sizeof(T));
                src_tensor.SetGlobalBuffer(data_addr, base_per_core);
                dst_tensor.SetGlobalBuffer(data_addr, base_per_core);
                aclshmemx_sdma_put_nbi(dst_tensor, src_tensor, tmp_local, base_per_core, i, EVENT_ID0);
            } else {
                __gm__ T* data_addr =
                    reinterpret_cast<__gm__ T*>(gva + i * elem_size * sizeof(T) + data_offset * sizeof(T));
                src_tensor.SetGlobalBuffer(data_addr, base_per_core);
                dst_tensor.SetGlobalBuffer(data_addr, base_per_core);
                aclshmemx_sdma_get_nbi(dst_tensor, src_tensor, tmp_local, base_per_core, i, EVENT_ID0);
            }
        }
        aclshmemx_sdma_quiet(tmp_local, EVENT_ID0);
    }
}

template <class T>
void allgather_kernel(uint32_t block_dim, void *stream, uint8_t *gva, int n_elements, uint8_t *device_dump,
    bool test_tensor_mode, bool is_put)
{
    if (!test_tensor_mode) {
        allgather_sdma<T><<<block_dim, nullptr, stream>>>(gva, n_elements, device_dump, is_put);
    } else {
        allgather_sdma_tensor<T><<<block_dim, nullptr, stream>>>(gva, n_elements, device_dump, is_put);
    }
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

template <class T>
int test_allgather_sdma(int my_pe, int n_pes)
{
    // ACLStream init
    aclrtStream stream = nullptr;
    CHECK_RET(aclrtCreateStream(&stream));

    constexpr uint32_t n_blocks = 20;
    constexpr int num10 = 10;

    uint8_t *device_dump = nullptr;
#if defined(ENABLE_ASCENDC_DUMP)
    CHECK_RET(aclrtMalloc(reinterpret_cast<void **>(&device_dump), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
#endif

    void *gva = aclshmem_malloc((128 * 1024 * 1024) * sizeof(T));

    // 初始化数据
    size_t trans_size = 16 * 1024 * 1024;
    std::vector<T> input(trans_size, 0);
    for (size_t i = 0; i < trans_size; i++) {
        input[i] = (T)(my_pe + num10);
    }

    CHECK_RET(aclrtMemcpy(reinterpret_cast<uint8_t *>(gva) + aclshmem_my_pe() * trans_size * sizeof(T),
        trans_size * sizeof(T), input.data(), trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));

    allgather_kernel<T>(n_blocks, stream, reinterpret_cast<uint8_t *>(gva), trans_size, device_dump, false, true);

    CHECK_RET(aclrtSynchronizeStream(stream));
    aclshmem_barrier_all();

#if defined(ENABLE_ASCENDC_DUMP)
    Adx::AdumpPrintWorkSpace(device_dump, ALL_DUMPSIZE, stream, "test");
#endif

    // 结果校验
    T* y_host;
    size_t input_size = n_pes * trans_size * sizeof(T);
    CHECK_RET(aclrtMallocHost(reinterpret_cast<void**>(&y_host), input_size));
    CHECK_RET(aclrtMemcpy(y_host, input_size, gva, input_size, ACL_MEMCPY_DEVICE_TO_HOST));

    const int check_step = 1; // 数据校验的步长
    for (int i = 0; i < n_pes; i++) {
        for (int j = 0; j < trans_size; j+= check_step) {
            int y = (int)(y_host[trans_size * i + j]);
            if (y != i + num10) {
                printf("ERROR in pe%d:%d %d != %d\n", i, j, y, i + num10);
                break;
            }
        }
    }

    CHECK_RET(aclrtFreeHost(y_host));
    aclshmem_free(gva);

    std::cout << " Pe " << my_pe << "Finished !! Result Correct !!" << std::endl;

    CHECK_RET(aclrtDestroyStream(stream));
    return 0;
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
        CHECK_RET(test_allgather_sdma<int>(my_pe, n_pes));
    } else if (std::string(data_type) == "uint8") {
        CHECK_RET(test_allgather_sdma<uint8_t>(my_pe, n_pes));
    } else if (std::string(data_type) == "int64") {
        CHECK_RET(test_allgather_sdma<int64_t>(my_pe, n_pes));
    } else if (std::string(data_type) == "fp32") {
        CHECK_RET(test_allgather_sdma<float>(my_pe, n_pes));
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