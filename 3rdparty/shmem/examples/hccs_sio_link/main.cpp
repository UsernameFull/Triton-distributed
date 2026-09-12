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
#include <cstdint>
#include <climits>
#include <memory>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"
#include "param.h"
#include "hccs_sio_link_kernel.h"

const char *ipport = "tcp://127.0.0.1:8998";
int f_pe = 0;
int f_npu = 0;
const char *data_type = "int";

aclshmemx_uniqueid_t default_flag_uid;

extern "C" aclError aclrtMemMapSelectedLink(void *virPtr2, size_t size, void *virPtr1, uint32_t linkIdx2);

/**
 * Setup HCCS link mapping: get heap base -> translate to remote address -> reserve VA -> map via HCCS link.
 * The resulting hccs_ptr provides a new VA that accesses the same PA as the SIO path but via HCCS link.
 */
static bool setup_hccs_mapping(int peer, uint64_t local_mem_size, void **hccs_ptr)
{
    int pe_id = aclshmem_my_pe();
    void *local_heap_base = aclshmemx_get_heap_base();
    if (local_heap_base == nullptr) {
        std::cerr << "PE " << pe_id << ": aclshmemx_get_heap_base failed" << std::endl;
        return false;
    }

    void *peer_heap_base = aclshmem_ptr(local_heap_base, peer);
    if (peer_heap_base == nullptr) {
        std::cerr << "PE " << pe_id << ": aclshmem_ptr failed for PE " << peer << std::endl;
        return false;
    }

    size_t heap_map_size = local_mem_size + ACLSHMEM_EXTRA_SIZE;

    aclError status = aclrtReserveMemAddress(hccs_ptr, heap_map_size, 0, nullptr, 0);
    if (status != ACL_ERROR_NONE || *hccs_ptr == nullptr) {
        std::cerr << "PE " << pe_id << ": aclrtReserveMemAddress failed, status: " << status << std::endl;
        return false;
    }

    status = aclrtMemMapSelectedLink(*hccs_ptr, heap_map_size, peer_heap_base, ACL_RT_MEM_LINK_IDX_1);
    if (status != ACL_ERROR_NONE) {
        std::cerr << "PE " << pe_id << ": aclrtMemMapSelectedLink failed for PE " << peer
                  << ", status: " << status << std::endl;
        return false;
    }
    return true;
}

static void teardown_hccs_mapping(void *hccs_ptr, bool hccs_mapped)
{
    int pe_id = aclshmem_my_pe();
    if (hccs_mapped) {
        aclError status = aclrtUnmapMem(hccs_ptr);
        if (status != ACL_ERROR_NONE) {
            std::cerr << "PE " << pe_id << ": aclrtUnmapMem failed, status: " << status << std::endl;
        }
    }
    if (hccs_ptr != nullptr) {
        aclError status = aclrtReleaseMemAddress(hccs_ptr);
        if (status != ACL_ERROR_NONE) {
            std::cerr << "PE " << pe_id << ": aclrtReleaseMemAddress failed, status: " << status << std::endl;
        }
    }
}

struct ShmemFree {
    void operator()(void *p) const { if (p) aclshmem_free(p); }
};
using ShmemUniquePtr = std::unique_ptr<void, ShmemFree>;

struct AclFreeHost {
    void operator()(void *p) const { if (p) aclrtFreeHost(p); }
};
using HostUniquePtr = std::unique_ptr<void, AclFreeHost>;

class AclStreamGuard {
public:
    AclStreamGuard() : stream_(nullptr) {}
    ~AclStreamGuard() { reset(); }
    aclrtStream get() const { return stream_; }
    aclrtStream* addr() { return &stream_; }
    void reset() {
        if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
    }
private:
    aclrtStream stream_;
};

struct HccsMappingGuard {
    void *ptr = nullptr;
    bool mapped = false;
    ~HccsMappingGuard() { teardown_hccs_mapping(ptr, mapped); }
};

#define CHECK_RET(func)                                                                                                \
    do {                                                                                                               \
        int ret = func;                                                                                                \
        if (ret != 0) {                                                                                                \
            std::cerr << __FILE__ << ":" << __LINE__ << " error: " << ret << std::endl;                                \
            return ret;                                                                                                \
        }                                                                                                              \
    } while (0)

int get_peer_pe(int pe_id, int n_pes)
{
    int peer = (pe_id % 2 == 0) ? (pe_id + 1) : (pe_id - 1);
    if (peer < 0 || peer >= n_pes) {
        return -1;
    }
    return peer;
}

template <class T>
int alloc_and_init_symmem(int pe_id, size_t data_elements, void *&local_addr, T *&init_host)
{
    size_t alloc_bytes = data_elements * sizeof(T);

    local_addr = aclshmem_malloc(alloc_bytes);
    if (local_addr == nullptr) {
        std::cerr << "PE " << pe_id << ": aclshmem_malloc failed" << std::endl;
        return -1;
    }

    aclError malloc_ret = aclrtMallocHost(reinterpret_cast<void **>(&init_host), alloc_bytes);
    if (malloc_ret != ACL_ERROR_NONE || init_host == nullptr) {
        std::cerr << "PE " << pe_id << ": aclrtMallocHost failed" << std::endl;
        aclshmem_free(local_addr);
        return -1;
    }
    for (size_t i = 0; i < data_elements; i++) {
        init_host[i] = static_cast<T>(pe_id * 1000 + i % 1000);
    }
    aclError memcpy_ret = aclrtMemcpy(local_addr, alloc_bytes, init_host, alloc_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (memcpy_ret != ACL_ERROR_NONE) {
        std::cerr << "PE " << pe_id << ": aclrtMemcpy host to device failed" << std::endl;
        aclrtFreeHost(init_host);
        aclshmem_free(local_addr);
        return -1;
    }
    return 0;
}

template <class T>
bool verify_buffer(int pe_id, int peer, void *buf, size_t data_elements, const char *tag, size_t start_index = 0)
{
    size_t alloc_bytes = data_elements * sizeof(T);

    T *result_host = nullptr;
    aclError malloc_ret = aclrtMallocHost(reinterpret_cast<void **>(&result_host), alloc_bytes);
    if (malloc_ret != ACL_ERROR_NONE || result_host == nullptr) {
        std::cerr << "PE " << pe_id << ": aclrtMallocHost failed in verify_buffer" << std::endl;
        return false;
    }
    aclError memcpy_ret = aclrtMemcpy(result_host, alloc_bytes, buf, alloc_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (memcpy_ret != ACL_ERROR_NONE) {
        std::cerr << "PE " << pe_id << ": [" << tag << "] aclrtMemcpy failed" << std::endl;
        aclrtFreeHost(result_host);
        return false;
    }

    bool correct = true;
    for (size_t i = 0; i < data_elements; i++) {
        T expected = static_cast<T>(peer * 1000 + (start_index + i) % 1000);
        if (static_cast<float>(result_host[i]) != static_cast<float>(expected)) {
            std::cerr << "PE " << pe_id << ": [" << tag << "] data mismatch at index " << (start_index + i)
                      << " from PE " << peer << ": got " << static_cast<float>(result_host[i])
                      << ", expected " << static_cast<float>(expected) << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "PE " << pe_id << ": [" << tag << "] path verification PASSED for PE " << peer << std::endl;
    }
    aclrtFreeHost(result_host);
    return correct;
}

template <class T>
bool copy_and_verify(int pe_id, int peer, void *symm_addr, size_t data_elements, const char *tag,
                     size_t start_index = 0, void *hccs_base = nullptr)
{
    size_t alloc_bytes = data_elements * sizeof(T);
    int elements = static_cast<int>(data_elements);

    void *local_buf = aclshmem_malloc(alloc_bytes);
    if (local_buf == nullptr) {
        std::cerr << "PE " << pe_id << ": [" << tag << "] aclshmem_malloc failed" << std::endl;
        return false;
    }

    aclrtStream stream = nullptr;
    aclError stream_ret = aclrtCreateStream(&stream);
    if (stream_ret != ACL_ERROR_NONE || stream == nullptr) {
        std::cerr << "PE " << pe_id << ": [" << tag << "] aclrtCreateStream failed" << std::endl;
        aclshmem_free(local_buf);
        return false;
    }

    constexpr uint32_t block_dim = 20;
    constexpr int ub_size_kb = 16;
    if (hccs_base != nullptr) {
        void *heap_base = aclshmemx_get_heap_base();
        uint64_t offset = reinterpret_cast<uintptr_t>(symm_addr) -
                           reinterpret_cast<uintptr_t>(heap_base);
        uint8_t *hccs_src = reinterpret_cast<uint8_t *>(
            reinterpret_cast<uintptr_t>(hccs_base) + offset);
        launch_mte_get_hccs<T>(block_dim, stream, reinterpret_cast<uint8_t *>(local_buf),
                                hccs_src, elements, ub_size_kb);
    } else {
        launch_mte_get<T>(block_dim, stream, reinterpret_cast<uint8_t *>(local_buf),
                                   reinterpret_cast<uint8_t *>(symm_addr), elements, peer, ub_size_kb);
    }
    aclError sync_ret = aclrtSynchronizeStream(stream);
    if (sync_ret != ACL_ERROR_NONE) {
        std::cerr << "PE " << pe_id << ": [" << tag << "] aclrtSynchronizeStream failed, ret=" << sync_ret << std::endl;
        aclrtDestroyStream(stream);
        aclshmem_free(local_buf);
        return false;
    }

    bool correct = verify_buffer<T>(pe_id, peer, local_buf, data_elements, tag, start_index);

    aclrtDestroyStream(stream);
    aclshmem_free(local_buf);
    return correct;
}

template <class T>
int test_sio_path(int pe_id, int n_pes, size_t data_elements)
{
    void *local_addr = nullptr;
    T *init_host = nullptr;
    if (alloc_and_init_symmem<T>(pe_id, data_elements, local_addr, init_host) != 0) {
        return -1;
    }
    ShmemUniquePtr local_addr_guard(local_addr);
    HostUniquePtr init_host_guard(init_host);

    int peer = get_peer_pe(pe_id, n_pes);
    if (peer < 0) {
        std::cerr << "PE " << pe_id << ": [SIO] no valid SIO peer" << std::endl;
        return -1;
    }

    bool correct = copy_and_verify<T>(pe_id, peer, local_addr, data_elements, "SIO");

    aclshmem_barrier_all();
    return correct ? 0 : -1;
}

template <class T>
int test_hccs_path(int pe_id, int n_pes, size_t data_elements, uint64_t local_mem_size)
{
    void *local_addr = nullptr;
    T *init_host = nullptr;
    if (alloc_and_init_symmem<T>(pe_id, data_elements, local_addr, init_host) != 0) {
        return -1;
    }
    ShmemUniquePtr local_addr_guard(local_addr);
    HostUniquePtr init_host_guard(init_host);

    int peer = get_peer_pe(pe_id, n_pes);
    if (peer < 0) {
        std::cerr << "PE " << pe_id << ": [HCCS] no valid HCCS peer" << std::endl;
        return -1;
    }

    HccsMappingGuard hccs;
    hccs.mapped = setup_hccs_mapping(peer, local_mem_size, &hccs.ptr);
    if (!hccs.mapped) {
        return -1;
    }

    int result = copy_and_verify<T>(pe_id, peer, local_addr, data_elements, "HCCS", 0, hccs.ptr) ? 0 : -1;

    aclshmem_barrier_all();
    return result;
}

template <class T>
int test_mixed_path(int pe_id, int n_pes, size_t data_elements, uint64_t local_mem_size)
{
    size_t sio_elements = data_elements * 3 / 5;
    size_t hccs_elements = data_elements - sio_elements;
    size_t sio_bytes = sio_elements * sizeof(T);
    size_t hccs_bytes = hccs_elements * sizeof(T);

    void *local_addr = nullptr;
    T *init_host = nullptr;
    if (alloc_and_init_symmem<T>(pe_id, data_elements, local_addr, init_host) != 0) {
        return -1;
    }
    ShmemUniquePtr local_addr_guard(local_addr);
    HostUniquePtr init_host_guard(init_host);

    void *hccs_symm_addr = reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(local_addr) + sio_bytes);

    int peer = get_peer_pe(pe_id, n_pes);
    if (peer < 0) {
        std::cerr << "PE " << pe_id << ": [MIXED] no valid peer" << std::endl;
        return -1;
    }

    HccsMappingGuard hccs;
    hccs.mapped = setup_hccs_mapping(peer, local_mem_size, &hccs.ptr);
    if (!hccs.mapped) {
        return -1;
    }

    void *heap_base = aclshmemx_get_heap_base();
    uint64_t hccs_offset = reinterpret_cast<uintptr_t>(hccs_symm_addr) -
                           reinterpret_cast<uintptr_t>(heap_base);
    void *hccs_src = reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(hccs.ptr) + hccs_offset);

    ShmemUniquePtr sio_buf(aclshmem_malloc(sio_bytes));
    if (sio_buf.get() == nullptr) {
        std::cerr << "PE " << pe_id << ": [MIXED] aclshmem_malloc for SIO buf failed" << std::endl;
        return -1;
    }

    ShmemUniquePtr hccs_buf(aclshmem_malloc(hccs_bytes));
    if (hccs_buf.get() == nullptr) {
        std::cerr << "PE " << pe_id << ": [MIXED] aclshmem_malloc for HCCS buf failed" << std::endl;
        return -1;
    }

    AclStreamGuard stream;
    aclError ret = aclrtCreateStream(stream.addr());
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "PE " << pe_id << ": [MIXED] aclrtCreateStream failed" << std::endl;
        return -1;
    }

    constexpr uint32_t block_dim = 20;
    constexpr int ub_size_kb = 16;
    int sio_elements_int = static_cast<int>(sio_elements);
    int hccs_elements_int = static_cast<int>(hccs_elements);

    launch_mte_get_mixed<T>(block_dim, stream.get(),
                             reinterpret_cast<uint8_t *>(sio_buf.get()),
                             reinterpret_cast<uint8_t *>(local_addr), sio_elements_int, peer,
                             reinterpret_cast<uint8_t *>(hccs_buf.get()),
                             reinterpret_cast<uint8_t *>(hccs_src), hccs_elements_int,
                             ub_size_kb);

    ret = aclrtSynchronizeStream(stream.get());
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "PE " << pe_id << ": [MIXED] aclrtSynchronizeStream failed, ret=" << ret << std::endl;
        return -1;
    }

    {
        bool sio_correct = verify_buffer<T>(pe_id, peer, sio_buf.get(), sio_elements, "MIXED-SIO");
        bool hccs_correct = verify_buffer<T>(pe_id, peer, hccs_buf.get(), hccs_elements, "MIXED-HCCS", sio_elements);
        int result = (sio_correct && hccs_correct) ? 0 : -1;
        aclshmem_barrier_all();
        return result;
    }
}

int main(int argc, char *argv[])
{
    if (argc <= INDEX8) {
        std::cerr << "Usage: " << argv[0]
                  << " <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> <data_type> <link_mode> [data_size_kb]"
                  << std::endl;
        return -1;
    }

    int n_pes = atoi(argv[INDEX1]);
    int pe_id = atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    f_pe = atoi(argv[INDEX5]);
    f_npu = atoi(argv[INDEX6]);
    data_type = argv[INDEX7];
    const char *link_mode = argv[INDEX8];
    size_t data_size_kb = (argc > INDEX9) ? static_cast<size_t>(atoi(argv[INDEX9])) : 4;

    if (n_pes <= 0 || pe_id < 0 || pe_id >= n_pes) {
        std::cerr << "Invalid arguments: n_pes must be > 0, pe_id must be in [0, n_pes)" << std::endl;
        std::cerr << "Usage: " << argv[0]
                  << " <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu> <data_type> <link_mode> [data_size_kb]"
                  << std::endl;
        return -1;
    }

    int32_t device_id = pe_id % n_pes + f_npu;
    CHECK_RET(aclInit(nullptr));
    CHECK_RET(aclrtSetDevice(device_id));

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    CHECK_RET(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    size_t type_size;
    if (std::string(data_type) == "int") {
        type_size = sizeof(int);
    } else if (std::string(data_type) == "int64") {
        type_size = sizeof(int64_t);
    } else {
        type_size = sizeof(float);
    }
    size_t data_elements = (data_size_kb * 1024UL) / type_size;

    if (data_size_kb == 0 || data_size_kb > SIZE_MAX / 1024UL || data_elements > static_cast<size_t>(INT_MAX)) {
        std::cerr << "Invalid data_size_kb: " << data_size_kb
                  << ", must be > 0 and result in data_elements <= " << INT_MAX << std::endl;
        return -1;
    }

    int status = 0;

    std::cout << "PE " << pe_id << ": link_mode=" << link_mode
              << ", data_size=" << data_size_kb << "KB, data_elements=" << data_elements << std::endl;

    if (std::string(link_mode) == "sio") {
        if (std::string(data_type) == "int") {
            status = test_sio_path<int>(pe_id, n_pes, data_elements);
        } else if (std::string(data_type) == "int64") {
            status = test_sio_path<int64_t>(pe_id, n_pes, data_elements);
        } else if (std::string(data_type) == "fp32") {
            status = test_sio_path<float>(pe_id, n_pes, data_elements);
        } else {
            std::cerr << "Unsupported data_type: " << data_type << std::endl;
            status = -1;
        }
    } else if (std::string(link_mode) == "hccs") {
        if (std::string(data_type) == "int") {
            status = test_hccs_path<int>(pe_id, n_pes, data_elements, local_mem_size);
        } else if (std::string(data_type) == "int64") {
            status = test_hccs_path<int64_t>(pe_id, n_pes, data_elements, local_mem_size);
        } else if (std::string(data_type) == "fp32") {
            status = test_hccs_path<float>(pe_id, n_pes, data_elements, local_mem_size);
        } else {
            std::cerr << "Unsupported data_type: " << data_type << std::endl;
            status = -1;
        }
    } else if (std::string(link_mode) == "all") {
        int sio_status = 0;
        int hccs_status = 0;
        if (std::string(data_type) == "int") {
            sio_status = test_sio_path<int>(pe_id, n_pes, data_elements);
            hccs_status = test_hccs_path<int>(pe_id, n_pes, data_elements, local_mem_size);
        } else if (std::string(data_type) == "int64") {
            sio_status = test_sio_path<int64_t>(pe_id, n_pes, data_elements);
            hccs_status = test_hccs_path<int64_t>(pe_id, n_pes, data_elements, local_mem_size);
        } else if (std::string(data_type) == "fp32") {
            sio_status = test_sio_path<float>(pe_id, n_pes, data_elements);
            hccs_status = test_hccs_path<float>(pe_id, n_pes, data_elements, local_mem_size);
        } else {
            std::cerr << "Unsupported data_type: " << data_type << std::endl;
            status = -1;
        }
        if (sio_status != 0 || hccs_status != 0) {
            status = -1;
        }
    } else if (std::string(link_mode) == "mixed") {
        if (std::string(data_type) == "int") {
            status = test_mixed_path<int>(pe_id, n_pes, data_elements, local_mem_size);
        } else if (std::string(data_type) == "int64") {
            status = test_mixed_path<int64_t>(pe_id, n_pes, data_elements, local_mem_size);
        } else if (std::string(data_type) == "fp32") {
            status = test_mixed_path<float>(pe_id, n_pes, data_elements, local_mem_size);
        } else {
            std::cerr << "Unsupported data_type: " << data_type << std::endl;
            status = -1;
        }
    } else {
        std::cerr << "Unknown link_mode: " << link_mode
                  << ". Supported: sio, hccs, all, mixed" << std::endl;
        status = -1;
    }

    CHECK_RET(aclshmem_finalize());
    CHECK_RET(aclrtResetDevice(device_id));
    CHECK_RET(aclFinalize());

    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] hccs_sio_link run success in pe " << pe_id << std::endl;
    return 0;
}
