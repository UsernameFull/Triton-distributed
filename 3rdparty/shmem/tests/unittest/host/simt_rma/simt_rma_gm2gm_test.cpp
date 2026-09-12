/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file simt_rma_gm2gm_test.cpp
 * @brief simt rma gm2gm unit tests host implementation
 *
 * This file tests the following functions:
 *      
 * |========SingleThread=======||=========Warp(32Threads)=========||========Block(AllThreads)=========|
 * |--------------------------------------Put blocking------------------------------------------------|
 * | aclshmem_##NAME##_put     || aclshmemx_##NAME##_put_warp     || aclshmemx_##NAME##_put_block     |
 * | aclshmem_put##BITS        || aclshmemx_put##BITS##_warp      || aclshmemx_put##BITS##_block      |
 * | aclshmem_putmem           || aclshmemx_putmem_warp           || aclshmemx_putmem_block           |
 * | aclshmem_##NAME##_p                                                                              |
 * |-------------------------------------Put NonBlocking----------------------------------------------|
 * | aclshmem_##NAME##_put_nbi || aclshmemx_##NAME##_put_nbi_warp || aclshmemx_##NAME##_put_nbi_block |
 * | aclshmem_put##BITS##_nbi  || aclshmemx_put##BITS##_nbi_warp  || aclshmemx_put##BITS##_nbi_block  |
 * | aclshmem_putmem_nbi       || aclshmemx_putmem_nbi_warp       || aclshmemx_putmem_nbi_block       |
 * |--------------------------------------Get blocking------------------------------------------------|
 * | aclshmem_##NAME##_get     || aclshmemx_##NAME##_get_warp     || aclshmemx_##NAME##_get_block     |
 * | aclshmem_get##BITS        || aclshmemx_get##BITS##_warp      || aclshmemx_get##BITS##_block      |
 * | aclshmem_getmem           || aclshmemx_getmem_warp           || aclshmemx_getmem_block           |
 * | aclshmem_##NAME##_g                                                                              |
 * |-------------------------------------Get NonBlocking----------------------------------------------|
 * | aclshmem_##NAME##_get_nbi || aclshmemx_##NAME##_get_nbi_warp || aclshmemx_##NAME##_get_nbi_block |
 * | aclshmem_get##BITS##_nbi  || aclshmemx_get##BITS##_nbi_warp  || aclshmemx_get##BITS##_nbi_block  |
 * | aclshmem_getmem_nbi       || aclshmemx_getmem_nbi_warp       || aclshmemx_getmem_nbi_block       |
 * |==================================================================================================|
 *
 * Choices of ##NAME##, ##BITS##:
 *      ##NAME##: {half, float, int8, int16, int32, int64, uint8, uint16, uint32, uint64, char, char, bfloat16}
 *      ##BITS##: {8, 16, 32, 64, 128}
 * All functions (except scalar functions xxx_g and xxx_p) share the same interface: __simt_callee__ inline void function_name(void *dst, void *src, size_t elem_size, int32_t pe)
 *
 * Interfaces of scalar functions:
 *      __simt_callee__ inline void aclshmem_##NAME##_p(TYPE *dst, const TYPE value, int pe)
 *      __simt_callee__ inline TYPE aclshmem_##NAME##_g(TYPE *src, int32_t pe)
 * For simplicity, scalar functions are wrapped into their corresponding non-scalar functions.
 * 
 * Each test case launches two processes, representing two PEs.
 * Each PE initializes two data buffers (one in host memory and one in device symmetric memory) with the value of its PE ID (0 or 1).
 * For Get operations, the first PE transfers data from the symmetric buffer of the second PE to its own symmetric buffer.
 * For Put operations, the first PE transfers data from its symmetric buffer to the symmetric buffer of the second PE.
 * Host memory is used to initialize and verify the data.
 */

#if defined(USE_SIMT)

#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <cstring>
#include <tuple>
#include <variant>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"
#include "simt_rma_gm2gm_definitions.h"

using InputParams = std::tuple<void*, void*, size_t, int32_t>; // void *dst, void *src, size_t elem_size, int32_t pe

struct InputParamsTransform {
    InputParams operator()(TypedSig sig, InputParams input_params)
    {
        auto [dst, src, elem_size, pe] = input_params;
        return InputParams(dst, src, elem_size / get_type_size(sig.name), pe);
    }

    InputParams operator()(SizedSig sig, InputParams input_params)
    {
        auto [dst, src, elem_size, pe] = input_params;
        return InputParams(dst, src, elem_size / get_bits_size(sig.bits), pe);
    }

    InputParams operator()(MemorySig sig, InputParams input_params)
    {
        return input_params;
    }

    InputParams operator()(ScalarSig sig, InputParams input_params)
    {
        auto [dst, src, elem_size, pe] = input_params;
        return InputParams(dst, src, 1, pe);
    }
};

void init_mem(void* mem, size_t size, int8_t pe)
{
    int8_t* mem_ptr = static_cast<int8_t*>(mem);
    for (size_t i = 0; i < size; ++i) {
        mem_ptr[i] = pe;
    }
}

bool check_mem(void* mem, size_t size, int8_t target_pe)
{
    int8_t* mem_ptr = static_cast<int8_t*>(mem);
    for (size_t i = 0; i < size; ++i) {
        if (mem_ptr[i] != target_pe) {
            return false;
        }
    }
    return true;
}

int32_t get_check_size(Sigs sig, int32_t default_value)
{
    return std::visit([default_value](auto&& s) -> int32_t {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, ScalarSig>) {
            return get_type_size(s.name);
        } else {
            return default_value;
        }
    }, sig);
}

int32_t get_peer_pe(int32_t my_pe, int32_t n_pes)
{
    return (my_pe + 1) % n_pes;
}

int32_t get_checking_pe(Sigs sig, int32_t my_pe, int32_t n_pes)
{
    OpType op = std::visit([](const auto& s) { return s.op; }, sig);
    if (op == OpType::GET) {
        return 0;
    }
    return 1;
}


inline void print_array_bytes(const void* arr, size_t len, const std::string& prefix = "")
{
    const unsigned char* p = static_cast<const unsigned char*>(arr);
    std::cout << prefix;
    for (size_t i = 0; i < len; ++i) {
        if (i) std::cout << ' ';
        std::cout << static_cast<int>(p[i]);
    }
    std::cout << '\n';
}

void invoke_test(FunctionInterface* func, Sigs sig, InputParams input_params, aclrtStream stream)
{
    auto input_param = std::visit([&](auto&& s) {
        return InputParamsTransform()(s, input_params);
    }, sig);
    auto [dst, src, elem_size, pe] = input_param;
    func(stream, dst, src, elem_size, pe);
}

inline std::ostream& operator<<(std::ostream& os, const TestCaseEntry& t) {
    os << "{func, ";
    const Sigs& sig = std::get<1>(t);
    std::visit([&os](auto const& s) { os << s; }, sig);
    os << "}";
    return os;
}

class SimtRMATest : public ::testing::TestWithParam<TestCaseEntry> {};

static TestCaseEntry g_current_test_case_entry = Entries[0];

void test_simt_rma_gm2gm(int my_pe, int n_pes, uint64_t local_mem_size)
{
    aclrtStream stream = nullptr;
    int32_t device_id = my_pe % test_gnpu_num + test_first_npu; 
    test_init(my_pe, n_pes, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    constexpr int32_t mem_size = 4096 * sizeof(int8_t); // bytes
    
    void* host_mem = nullptr;
    void* device_mem = nullptr;
    
    ASSERT_EQ(aclrtMallocHost(&host_mem, mem_size), ACL_SUCCESS);
    device_mem = aclshmem_malloc(mem_size);
    ASSERT_NE(device_mem, nullptr);
    
    init_mem(host_mem, mem_size, my_pe);
    ASSERT_EQ(aclrtMemcpy(device_mem, mem_size, host_mem, mem_size, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    
    auto [func, sig] = g_current_test_case_entry;
    InputParams input_params = std::make_tuple(device_mem, device_mem, mem_size, 1);

    aclshmem_barrier_all();
    if (my_pe == 0) {
        invoke_test(func, sig, input_params, stream);
    }

    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmem_barrier_all();
    EXPECT_EQ(aclrtMemcpy(host_mem, mem_size, device_mem, mem_size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    if (my_pe == get_checking_pe(sig, my_pe, n_pes)) {
        int32_t check_size = get_check_size(sig, mem_size);
        EXPECT_TRUE(check_mem(host_mem, check_size, get_peer_pe(my_pe, n_pes)));
    }

    aclshmem_free(device_mem);
    aclrtFreeHost(host_mem);

    test_finalize(stream, device_id);
}

TEST_P(SimtRMATest, SimtRMAGm2GmTest) 
{
    constexpr int32_t process_count = 2;
    constexpr uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    g_current_test_case_entry = GetParam();
    test_mutil_task(test_simt_rma_gm2gm, local_mem_size, process_count);
}

INSTANTIATE_TEST_SUITE_P(SimtRMAGm2GmTestCases, SimtRMATest, ::testing::ValuesIn(Entries));

#endif // defined(ACLSHMEM_SIMT_SUPPORT)