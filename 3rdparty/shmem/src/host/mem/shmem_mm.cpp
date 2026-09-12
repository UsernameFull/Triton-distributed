/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmemi_mm.h"

int32_t memory_manager_initialize(void *base, uint64_t size, aclshmem_mem_type_t mem_type)
{
    if (mem_type == HOST_SIDE) {
        aclshmemi_host_memory_manager = std::make_shared<memory_manager>(base, size);
        return ACLSHMEM_SUCCESS;
    }
    aclshmemi_memory_manager = std::make_shared<memory_manager>(base, size);
    if (aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Failed to initialize shared memory heap");
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

void memory_manager_destroy()
{
    aclshmemi_memory_manager.reset();
    if (aclshmemi_host_memory_manager != nullptr) {
        aclshmemi_host_memory_manager.reset();
    }
}

void *aclshmem_malloc(size_t size)
{
    if (aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }

    void *ptr = aclshmemi_memory_manager->allocate(size);
    SHM_LOG_DEBUG("aclshmem_malloc(" << size << ")" << " ptr: " << ptr);
    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("malloc mem barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            aclshmemi_memory_manager->release(ptr);
            ptr = nullptr;
        }
    }
#ifdef DEBUG_MODE
    ret = is_alloc_size_symmetric(size);
    if (ret != 0) {
        SHM_LOG_ERROR("asymmetric alloc detected");
        return nullptr;
    }
#endif
    return ptr;
}

void *aclshmem_calloc(size_t nmemb, size_t size)
{
    if (aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }
    SHM_ASSERT_MULTIPLY_OVERFLOW(nmemb, size, g_state.heap_size, nullptr);

    auto total_size = nmemb * size;
    auto ptr = aclshmemi_memory_manager->allocate(total_size);
    if (ptr != nullptr) {
        auto ret = aclrtMemset(ptr, size, 0, size);
        if (ret != 0) {
            SHM_LOG_ERROR("aclshmem_calloc(" << nmemb << ", " << size << ") memset failed: " << ret);
            aclshmemi_memory_manager->release(ptr);
            ptr = nullptr;
        }
    }

    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("calloc mem barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            aclshmemi_memory_manager->release(ptr);
            ptr = nullptr;
        }
    }

    SHM_LOG_DEBUG("aclshmem_calloc(" << nmemb << ", " << size << ")");
    return ptr;
}

void *aclshmem_align(size_t alignment, size_t size)
{
    if (aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }

    auto ptr = aclshmemi_memory_manager->aligned_allocate(alignment, size);
    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmem_align barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            aclshmemi_memory_manager->release(ptr);
            ptr = nullptr;
        }
    }
    SHM_LOG_DEBUG("aclshmem_align(" << alignment << ", " << size << ")");
    return ptr;
}

void aclshmem_free(void *ptr)
{
    if (aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return;
    }
    if (ptr == nullptr) {
        return;
    }

    auto ret = aclshmemi_memory_manager->release(ptr);
    if (ret != 0) {
        SHM_LOG_ERROR("release failed: " << ret);
    }

    SHM_LOG_DEBUG("aclshmem_free " << ret);
}

void *aclshmemx_get_heap_base(aclshmem_mem_type_t mem_type)
{
    if (mem_type == HOST_SIDE) {
        if (g_state.host_heap_base == nullptr) {
            SHM_LOG_ERROR("Host Memory Heap Not Initialized.");
            return nullptr;
        }
        return g_state.host_heap_base;
    }
    if (g_state.heap_base == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }
    return g_state.heap_base;
}

bool support_host_mem_type(aclshmem_mem_type_t mem_type)
{
#ifndef HAS_ACLRT_MEM_FABRIC_HANDLE
    if (mem_type == HOST_SIDE) {
        SHM_LOG_ERROR("Not support HOST_SIDE malloc, please update CANN version");
        return false;
    }
#endif
    return true;
}

std::shared_ptr<memory_manager> getory_manager(aclshmem_mem_type_t mem_type)
{
    if (mem_type == HOST_SIDE) {
        if (aclshmemi_host_memory_manager != nullptr) {
            return aclshmemi_host_memory_manager;
        }
        if (init_manager->setup_heap(HOST_SIDE)) {
            SHM_LOG_ERROR("Host Memory Heap Not Initialized.");
            return nullptr;
        }
        if (memory_manager_initialize(g_state.host_heap_base, g_state.heap_size, HOST_SIDE) != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("Host Memory Heap Not Initialized.");
            return nullptr;
        }
        if (init_manager->update_device_state((void *)&g_state, sizeof(aclshmem_device_host_state_t)) !=
            ACLSHMEM_SUCCESS) {
            return nullptr;
        }
        return aclshmemi_host_memory_manager;
    }
    if (aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }
    return aclshmemi_memory_manager;
}

void *aclshmemx_malloc(size_t size, aclshmem_mem_type_t mem_type)
{
    if (!support_host_mem_type(mem_type)) {
        return nullptr;
    }
    auto mem_manager = getory_manager(mem_type);
    if (mem_manager == nullptr) {
        return nullptr;
    }
    void *ptr = mem_manager->allocate(size);
    SHM_LOG_DEBUG("aclshmem_malloc(" << size << ")");
    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("malloc mem barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            mem_manager->release(ptr);
            ptr = nullptr;
        }
    }
    return ptr;
}

void *aclshmemx_calloc(size_t nmemb, size_t size, aclshmem_mem_type_t mem_type)
{
    if (!support_host_mem_type(mem_type)) {
        return nullptr;
    }
    auto mem_manager = getory_manager(mem_type);
    if (mem_manager == nullptr) {
        return nullptr;
    }
    SHM_ASSERT_MULTIPLY_OVERFLOW(nmemb, size, g_state.heap_size, nullptr);
    auto total_size = nmemb * size;
    auto ptr = mem_manager->allocate(total_size);
    if (ptr != nullptr) {
        auto ret = aclrtMemset(ptr, size, 0, size);
        if (ret != 0) {
            SHM_LOG_ERROR("aclshmem_calloc(" << nmemb << ", " << size << ") memset failed: " << ret);
            mem_manager->release(ptr);
            ptr = nullptr;
        }
    }
    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("calloc mem barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            mem_manager->release(ptr);
            ptr = nullptr;
        }
    }

    SHM_LOG_DEBUG("aclshmem_calloc(" << nmemb << ", " << size << ")");
    return ptr;
}

void *aclshmemx_align(size_t alignment, size_t size, aclshmem_mem_type_t mem_type)
{
    if (!support_host_mem_type(mem_type)) {
        return nullptr;
    }
    auto mem_manager = getory_manager(mem_type);
    if (mem_manager == nullptr) {
        return nullptr;
    }
    auto ptr = mem_manager->aligned_allocate(alignment, size);
    auto ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmem_align barrier failed, ret: " << ret);
        if (ptr != nullptr) {
            mem_manager->release(ptr);
            ptr = nullptr;
        }
    }
    SHM_LOG_DEBUG("aclshmem_align(" << alignment << ", " << size << ")");
    return ptr;
}

void aclshmemx_free(void *ptr, aclshmem_mem_type_t mem_type)
{
    if (!support_host_mem_type(mem_type)) {
        return;
    }
    if (ptr == nullptr) {
        return;
    }
    if (mem_type == HOST_SIDE && (aclshmemi_host_memory_manager == nullptr)) {
        SHM_LOG_ERROR("Host Memory Heap Not Initialized.");
        return;
    }
    if (mem_type == DEVICE_SIDE && aclshmemi_memory_manager == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return;
    }
    auto ret = mem_type == HOST_SIDE ? aclshmemi_host_memory_manager->release(ptr) : aclshmemi_memory_manager->release(ptr);;
    if (ret != 0) {
        SHM_LOG_ERROR("release failed: " << ret);
    }

    SHM_LOG_DEBUG("aclshmem_free " << ret);
}