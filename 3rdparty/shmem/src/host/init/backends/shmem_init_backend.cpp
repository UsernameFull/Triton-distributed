/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <random>
#include <algorithm>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include "shmemi_init_backend.h"
#include "shmemi_host_common.h"
#include "host/shmem_host_def.h"
#include "utils/shmemi_logger.h"
#include "store_factory.h"
#include "mem_entity_entry.h"
#include "sotre_net.h"
#include "store_net_common.h"
#include "mem_entity_def.h"

aclshmemi_init_backend::aclshmemi_init_backend()
{
    auto status = aclrtGetDevice(&device_id);
    if (status != 0) {
        SHM_LOG_ERROR("Get Device_id error");
    }

    mstx_reg_ptr_ = shm::create_mstx_mem_register_instance();
    if (mstx_reg_ptr_ == nullptr) {
        SHM_LOG_ERROR("create mstx_reg_ptr_ returned nullptr!");
    }
}

aclshmemi_init_backend::~aclshmemi_init_backend()
{
    if (mstx_reg_ptr_ != nullptr) {
        delete mstx_reg_ptr_;
        mstx_reg_ptr_ = nullptr;
    }
}

int aclshmemi_init_backend::bind_aclshmem_entity(aclshmemx_init_attr_t *attr, aclshmem_device_host_state_t *state, aclshmemi_bootstrap_handle_t *handle)
{
    std::lock_guard<std::mutex> lock(entity_map_mutex_);
    // fetch entity resources
    int instance_id = attr->instance_id;
    entity_member *elem = new entity_member;
    elem->entity_attr = attr;
    elem->entity_host_state = state;
    elem->entity_device_state = (aclshmem_device_host_state_t *)std::calloc(1, sizeof(aclshmem_device_host_state_t));
    if (elem->entity_device_state == nullptr) {
        SHM_LOG_ERROR("Failed to allocate memory for entity_device_state.");
        delete elem;
        return ACLSHMEM_INNER_ERROR;
    }
    elem->entity_boot_handle = handle;

    entity_map_[instance_id] = elem;

    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::release_aclshmem_entity(uint64_t instance_id)
{
    std::lock_guard<std::mutex> lock(entity_map_mutex_);
    // fetch entity_member
    auto it = entity_map_.find(instance_id);
    if (it == entity_map_.end() || it->second == nullptr) {
        SHM_LOG_ERROR("Inner backend find instance context failed !");
        return ACLSHMEM_INNER_ERROR;
    }
    entity_member *elem = it->second;

    if (elem->entity_host_state != nullptr) {
        elem->entity_host_state->is_aclshmem_created = false;
    }

    if (elem->entity_device_state != nullptr) {
        std::free(elem->entity_device_state);
        elem->entity_device_state = nullptr;
    }

    entity_map_.erase(it);
    delete elem;

    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::init_device_state()
{
    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto it = entity_map_.find(g_instance_ctx->id);
        if (it == entity_map_.end() || it->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = it->second;
    }

    int ret = 0;
    // 初始化hybm，分配meta存储空间
    if (hybm_init_count == 0) {
        ret = hybm_init(device_id, 0);
        if (ret != 0) {
            SHM_LOG_ERROR("init hybm failed, result: " << ret);
            return ACLSHMEM_SMEM_ERROR;
        }

        mstx_reg_ptr_->add_mem_regions((void *)shm::HYBM_DEVICE_META_ADDR, shm::HYBM_DEVICE_INFO_SIZE, MSTX_GROUP_FOR_STATE);
        mstx_reg_ptr_->mstx_mem_regions_register(MSTX_GROUP_FOR_STATE);
    }
    hybm_init_count++;

    ret = aclshmemi_control_barrier_all();  // 保证所有rank都初始化了
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_control_barrier_all failed, result: " << ret);
        return ACLSHMEM_SMEM_ERROR;
    }

    SHM_LOG_INFO("init_device_state success. world_size: " << elem->entity_attr->n_pes);
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::create_entity(aclshmem_mem_type_t mem_type)
{
    uint64_t instance_id = g_instance_ctx->id;
    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto it = entity_map_.find(instance_id);
        if (it == entity_map_.end() || it->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = it->second;
    }

    auto attributes = elem->entity_attr;
    auto host_state = elem->entity_host_state;
    if (attributes == nullptr || host_state == nullptr) {
        SHM_LOG_ERROR("One of entity_member's required fields is null: "
                      "attributes, host_state. Please Check!");
        return ACLSHMEM_INNER_ERROR;
    }

    // 创建entity
    hybm_options options;
    options.bmType = HYBM_TYPE_AI_CORE_INITIATE;
    options.bmDataOpType = static_cast<hybm_data_op_type>(HYBM_DOP_TYPE_MTE);
    if (attributes->option_attr.data_op_engine_type & ACLSHMEM_DATA_OP_ROCE) {
        auto temp = static_cast<uint32_t>(options.bmDataOpType) | HYBM_DOP_TYPE_DEVICE_RDMA;
        options.bmDataOpType = static_cast<hybm_data_op_type>(temp);
    }
    if (attributes->option_attr.data_op_engine_type & ACLSHMEM_DATA_OP_SDMA) {
        auto temp = static_cast<uint32_t>(options.bmDataOpType) | HYBM_DOP_TYPE_DEVICE_SDMA;
        options.bmDataOpType = static_cast<hybm_data_op_type>(temp);
    }
    if (attributes->option_attr.data_op_engine_type & ACLSHMEM_DATA_OP_UDMA) {
        auto temp = static_cast<uint32_t>(options.bmDataOpType) | HYBM_DOP_TYPE_DEVICE_UDMA;
        options.bmDataOpType = static_cast<hybm_data_op_type>(temp);
    }
    options.bmScope = HYBM_SCOPE_CROSS_NODE;
    options.rankCount = attributes->n_pes;
    options.rankId = attributes->my_pe;
    if (mem_type == HOST_SIDE) {
        options.memType = HYBM_MEM_TYPE_HOST;
        options.deviceVASpace = 0;
        options.hostVASpace = host_state->heap_size;
    } else {
        options.memType = HYBM_MEM_TYPE_DEVICE;
        options.deviceVASpace = host_state->heap_size;
        options.hostVASpace = 0;
    }
    options.preferredGVA = 0;
    options.role = HYBM_ROLE_PEER;
    std::string defaultNic = "10002";
    std::copy_n(defaultNic.c_str(), defaultNic.size() + 1, options.nic);
    auto entity_id = instance_id << 1;
    if (mem_type == HOST_SIDE) {
        elem->dram_entity = hybm_create_entity(entity_id + 1, &options, 0);
        if (elem->dram_entity == nullptr) {
            SHM_LOG_ERROR("create host dram entity failed");
            return ACLSHMEM_SMEM_ERROR;
        }
        return ACLSHMEM_SUCCESS;
    }

    elem->hbm_entity = hybm_create_entity(entity_id, &options, 0);
    if (elem->hbm_entity == nullptr) {
        SHM_LOG_ERROR("create device hbm entity failed");
        return ACLSHMEM_SMEM_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::update_device_state(void* host_ptr, size_t size)
{
    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto it = entity_map_.find(g_instance_ctx->id);
        if (it == entity_map_.end() || it->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = it->second;
    }

    auto hbm_entity = elem->hbm_entity;
    auto host_state = elem->entity_host_state;
    auto device_state = elem->entity_device_state;
    if (hbm_entity == nullptr || host_state == nullptr || device_state == nullptr) {
        SHM_LOG_ERROR("One of entity_member's required fields is null: "
                      "hbm_entity, host_state, device_state. Please Check!");
        return ACLSHMEM_INNER_ERROR;
    }

    int32_t ptr_size = host_state->npes * sizeof(void *);

    ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state->p2p_device_heap_base, ptr_size, host_state->p2p_device_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state->rdma_device_heap_base, ptr_size, host_state->rdma_device_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state->sdma_device_heap_base, ptr_size, host_state->sdma_device_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    if (host_state->host_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state->p2p_host_heap_base, ptr_size,
            host_state->p2p_host_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
        ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state->rdma_host_heap_base, ptr_size,
            host_state->rdma_host_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
        ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state->sdma_host_heap_base, ptr_size,
            host_state->sdma_host_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    auto backup_p2p = device_state->p2p_device_heap_base;
    auto backup_rdma = device_state->rdma_device_heap_base;
    auto backup_sdma = device_state->sdma_device_heap_base;
    auto backup_host_p2p = device_state->p2p_host_heap_base;
    auto backup_host_rdma = device_state->rdma_host_heap_base;
    auto backup_host_sdma = device_state->sdma_host_heap_base;
    std::memcpy(device_state, host_ptr, size);
    device_state->p2p_device_heap_base = backup_p2p;
    device_state->rdma_device_heap_base = backup_rdma;
    device_state->sdma_device_heap_base = backup_sdma;
    device_state->p2p_host_heap_base = backup_host_p2p;
    device_state->rdma_host_heap_base = backup_host_rdma;
    device_state->sdma_host_heap_base = backup_host_sdma;

    auto ret = hybm_set_extra_context(hbm_entity, device_state, size);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("hybm_set_extra_context failed, ret: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::finalize_device_state()
{
    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto it = entity_map_.find(g_instance_ctx->id);
        if (it == entity_map_.end() || it->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = it->second;
    }

    if (hybm_init_count == 0) {
        SHM_LOG_INFO("shm hybm backend not initialized yet, skip finalize device state.");
        return ACLSHMEM_SUCCESS;
    }

    hybm_init_count--;
    if (hybm_init_count == 0) {
        hybm_uninit();
    }

    SHM_LOG_INFO("shmemi uninit finished");
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::reserve_heap(aclshmem_mem_type_t mem_type)
{
    auto ret = create_entity(mem_type);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("create entity failed, result: " << ret << ", mem_type: " << mem_type << ".");
        return ret;
    }

    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto it = entity_map_.find(g_instance_ctx->id);
        if (it == entity_map_.end() || it->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = it->second;
    }

    auto entity = mem_type == HOST_SIDE ? elem->dram_entity : elem->hbm_entity;
    auto host_state = elem->entity_host_state;
    if (entity == nullptr || host_state == nullptr) {
        SHM_LOG_ERROR("One of entity_member's required fields is null: "
                      "entity, host_state. Please Check!");
        return ACLSHMEM_INNER_ERROR;
    }

    void *gva = nullptr;
    ret = hybm_reserve_mem_space(entity, 0, &gva);
    auto aligned = ALIGN_UP(host_state->heap_size, ACLSHMEM_HEAP_ALIGNMENT_SIZE);
    if (ret != 0 || gva == nullptr) {
        SHM_LOG_ERROR("reserve mem failed, result: " << ret << ", mem_type: " << mem_type << ".");
        return ACLSHMEM_SMEM_ERROR;
    }
    if (mem_type == HOST_SIDE) {
        elem->dram_gva = gva;
        mstx_reg_ptr_->add_mem_regions_multi_pe_align(elem->dram_gva, host_state->heap_size, aligned, host_state->npes, MSTX_GROUP_FOR_HOST_HEAP);
        mstx_reg_ptr_->mstx_mem_regions_register(MSTX_GROUP_FOR_HOST_HEAP);
    } else {
        elem->hbm_gva = gva;
        mstx_reg_ptr_->add_mem_regions_multi_pe_align(elem->hbm_gva, host_state->heap_size, aligned, host_state->npes, MSTX_GROUP_FOR_DEVICE_HEAP);
        mstx_reg_ptr_->mstx_mem_regions_register(MSTX_GROUP_FOR_DEVICE_HEAP);
    }
    SHM_LOG_INFO("reserve_heap success, mem_type: " << mem_type << ".");
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::reach_info_init(void *&gva)
{
    uint64_t instance_id = g_instance_ctx->id;
    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto it = entity_map_.find(instance_id);
        if (it == entity_map_.end() || it->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = it->second;
    }

    auto hbm_entity = elem->hbm_entity;
    auto host_state = elem->entity_host_state;
    if (hbm_entity == nullptr || host_state == nullptr) {
        SHM_LOG_ERROR("One of entity_member's required fields is null: "
                      "hbm_entity, host_state. Please Check!");
        return ACLSHMEM_INNER_ERROR;
    }

    auto aligned = ALIGN_UP(host_state->heap_size, ACLSHMEM_HEAP_ALIGNMENT_SIZE);

    for (int32_t i = 0; i < host_state->npes; i++) {
        hybm_data_op_type reaches_types;
        auto ret = hybm_entity_reach_types(hbm_entity, i, reaches_types, 0);
        if (ret != 0) {
            SHM_LOG_ERROR("hybm_entity_reach_types failed: " << ret);
            return ACLSHMEM_SMEM_ERROR;
        }
        host_state->p2p_device_heap_base[i] = (void *)((uintptr_t)gva + aligned * static_cast<uint32_t>(i));
        SHM_LOG_DEBUG("rank " << i << " p2p_heap_base: " << host_state->p2p_device_heap_base[i]);
        if (reaches_types & HYBM_DOP_TYPE_MTE) {
            host_state->topo_list[i] |= ACLSHMEM_TRANSPORT_MTE;
        }
        if (reaches_types & HYBM_DOP_TYPE_DEVICE_RDMA) {
            host_state->topo_list[i] |= ACLSHMEM_TRANSPORT_ROCE;
        }
        if (reaches_types & HYBM_DOP_TYPE_DEVICE_SDMA) {
            host_state->topo_list[i] |= ACLSHMEM_TRANSPORT_SDMA;
        }
        if (reaches_types & HYBM_DOP_TYPE_DEVICE_UDMA) {
            host_state->topo_list[i] |= ACLSHMEM_TRANSPORT_UDMA;
        }
        host_state->sdma_device_heap_base[i] = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::exchange_slice(aclshmem_mem_type_t mem_type)
{
    uint64_t instance_id = g_instance_ctx->id;
    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto it = entity_map_.find(instance_id);
        if (it == entity_map_.end() || it->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = it->second;
    }

    // export memory
    hybm_exchange_info ex_info;
    bzero(&ex_info, sizeof(ex_info));

    auto entity = mem_type == HOST_SIDE ? elem->dram_entity : elem->hbm_entity;
    auto slice = mem_type == HOST_SIDE ? elem->dram_slice : elem->hbm_slice;
    auto host_state = elem->entity_host_state;
    auto boot_handle = elem->entity_boot_handle;

    // silce is input, don't check
    if (entity == nullptr || host_state == nullptr || boot_handle == nullptr) {
        SHM_LOG_ERROR("One of entity_member's required fields is null: "
                      "entity, host_state, boot_handle. Please Check!");
        return ACLSHMEM_INNER_ERROR;
    }

    auto ret = hybm_export(entity, slice, 0, &ex_info);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm export slice failed, result: " << ret);
        return ret;
    }

    hybm_exchange_info all_ex_info[host_state->npes];
    ret = boot_handle->allgather((void *)&ex_info, (void *)all_ex_info, sizeof(hybm_exchange_info), boot_handle);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm gather export slice failed, result: " << ret);
        return ret;
    }

    // import memory
    ret = hybm_import(entity, all_ex_info, host_state->npes, nullptr, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm import failed, result: " << ret);
        return ret;
    }

    ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("hybm barrier for slice failed, result: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::exchange_entity(aclshmem_mem_type_t mem_type)
{
    uint64_t instance_id = g_instance_ctx->id;
    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto it = entity_map_.find(instance_id);
        if (it == entity_map_.end() || it->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = it->second;
    }

    // export entity
    auto entity = mem_type == HOST_SIDE ? elem->dram_entity : elem->hbm_entity;
    auto host_state = elem->entity_host_state;
    auto boot_handle = elem->entity_boot_handle;
    if (entity == nullptr || host_state == nullptr || boot_handle == nullptr) {
        SHM_LOG_ERROR("One of entity_member's required fields is null: "
                      "entity, host_state, boot_handle. Please Check!");
        return ACLSHMEM_INNER_ERROR;
    }

    hybm_exchange_info ex_info;
    bzero(&ex_info, sizeof(ex_info));
    auto ret = hybm_export(entity, nullptr, 0, &ex_info);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm export entity failed, result: " << ret);
        return ret;
    }

    if (ex_info.descLen == 0) {
        return ACLSHMEM_SUCCESS;
    }

    hybm_exchange_info all_ex_info[host_state->npes];
    ret = boot_handle->allgather((void *)&ex_info, (void *)all_ex_info, sizeof(hybm_exchange_info), boot_handle);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm gather export entity failed, result: " << ret);
        return ret;
    }
    // import entity
    ret = hybm_import(entity, all_ex_info, host_state->npes, nullptr, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm import entity failed, result: " << ret);
        return ret;
    }

    ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("hybm barrier for slice failed, result: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::setup_heap(aclshmem_mem_type_t mem_type)
{
    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto it = entity_map_.find(g_instance_ctx->id);
        if (it == entity_map_.end() || it->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = it->second;
    }

    auto entity = mem_type == HOST_SIDE ? elem->dram_entity : elem->hbm_entity;
    auto mType = mem_type == HOST_SIDE ? HYBM_MEM_TYPE_HOST : HYBM_MEM_TYPE_DEVICE;

    auto host_state = elem->entity_host_state;
    auto attributes = elem->entity_attr;
    auto device_state = elem->entity_device_state;
    if (entity == nullptr || host_state == nullptr || attributes == nullptr || device_state == nullptr) {
        SHM_LOG_ERROR("One of entity_member's required fields is null: "
                      "entity, host_state, attributes, device_state. Please Check!");
        return ACLSHMEM_INNER_ERROR;
    }

    // alloc memory
    auto slice = hybm_alloc_local_memory(entity, mType, host_state->heap_size, 0);
    if (slice == nullptr) {
        SHM_LOG_ERROR("alloc local mem failed, size: " << host_state->heap_size);
        return ACLSHMEM_SMEM_ERROR;
    }
    if (mem_type == HOST_SIDE) {
        elem->dram_slice = slice;
    } else {
        elem->hbm_slice = slice;
    }
    auto ret = exchange_slice(mem_type);
    if (ret != 0) {
        SHM_LOG_ERROR("exchange slice failed, result: " << ret);
        return ret;
    }
    ret = exchange_entity(mem_type);
    if (ret != 0) {
        SHM_LOG_ERROR("exchange entity failed, result: " << ret);
        return ret;
    }

    // mmap memory
    ret = hybm_mmap(entity, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm mmap failed, result: " << ret);
        return ret;
    }
    if (mem_type == HOST_SIDE) {
        auto aligned = ALIGN_UP(host_state->heap_size, ACLSHMEM_HEAP_ALIGNMENT_SIZE);
        host_state->host_heap_base = (void *)((uintptr_t)elem->dram_gva + aligned * static_cast<uint32_t>(attributes->my_pe));
        ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state->p2p_host_heap_base, host_state->npes * sizeof(void *)));
        ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state->rdma_host_heap_base, host_state->npes * sizeof(void *)));
        ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state->sdma_host_heap_base, host_state->npes * sizeof(void *)));

        ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state->p2p_host_heap_base, host_state->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
        ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state->rdma_host_heap_base, host_state->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
        ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state->sdma_host_heap_base, host_state->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
        for (int32_t i = 0; i < host_state->npes; i++) {
            host_state->p2p_host_heap_base[i] = (void *)((uintptr_t)elem->dram_gva + aligned * static_cast<uint32_t>(i));
        }
        SHM_LOG_DEBUG("host side dram heap setup successed, host_state->host_heap_base: " << (void *)(uintptr_t)host_state->host_heap_base);
        return ACLSHMEM_SUCCESS;
    }
    // device side heap info save
    auto aligned = ALIGN_UP(host_state->heap_size, ACLSHMEM_HEAP_ALIGNMENT_SIZE);
    host_state->heap_base = (void *)((uintptr_t)elem->hbm_gva + aligned * static_cast<uint32_t>(attributes->my_pe));
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state->p2p_device_heap_base, host_state->npes * sizeof(void *)));
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state->rdma_device_heap_base, host_state->npes * sizeof(void *)));
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state->sdma_device_heap_base, host_state->npes * sizeof(void *)));

    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state->p2p_device_heap_base, host_state->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state->rdma_device_heap_base, host_state->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state->sdma_device_heap_base, host_state->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    SHM_LOG_DEBUG("device side hbm heap setup successed, host_state->heap_base: " << (void *)(uintptr_t)host_state->heap_base << " p2p_device_heap_base: " << (void *)(uintptr_t)device_state->p2p_device_heap_base);
    ret = reach_info_init(elem->hbm_gva);
    if (ret != 0) {
        SHM_LOG_ERROR("reach_info_init failed, result: " << ret);
        return ret;
    }

    host_state->is_aclshmem_created = true;
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::remove_heap(aclshmem_mem_type_t mem_type)
{
    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto it = entity_map_.find(g_instance_ctx->id);
        if (it == entity_map_.end() || it->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = it->second;
    }

    auto entity = mem_type == HOST_SIDE ? elem->dram_entity : elem->hbm_entity;
    if (entity == nullptr) {
        SHM_LOG_DEBUG("entity not create, skip unmap.");
        return ACLSHMEM_SUCCESS;
    }

    auto host_state = elem->entity_host_state;
    auto device_state = elem->entity_device_state;
    if (host_state == nullptr || device_state == nullptr) {
        SHM_LOG_ERROR("One of entity_member's required fields is null: "
                      "host_state, device_state. Please Check!");
        return ACLSHMEM_INNER_ERROR;
    }

    if (host_state->p2p_device_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state->p2p_device_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state->p2p_device_heap_base));
        host_state->p2p_device_heap_base = nullptr;
    }
    if (host_state->rdma_device_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state->rdma_device_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state->rdma_device_heap_base));
        host_state->rdma_device_heap_base = nullptr;
    }
    if (host_state->sdma_device_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state->sdma_device_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state->sdma_device_heap_base));
        host_state->sdma_device_heap_base = nullptr;
    }
    if (host_state->p2p_host_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state->p2p_host_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state->p2p_host_heap_base));
        host_state->p2p_host_heap_base = nullptr;
    }
    if (host_state->rdma_host_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state->rdma_host_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state->rdma_host_heap_base));
        host_state->rdma_host_heap_base = nullptr;
    }
    if (host_state->sdma_host_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(host_state->sdma_host_heap_base));
        ACLSHMEM_CHECK_RET(aclrtFree(device_state->sdma_host_heap_base));
        host_state->sdma_host_heap_base = nullptr;
    }

    hybm_unmap(entity, 0);
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::release_heap(aclshmem_mem_type_t mem_type)
{
    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto it = entity_map_.find(g_instance_ctx->id);
        if (it == entity_map_.end() || it->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = it->second;
    }

    auto entity = mem_type == HOST_SIDE ? elem->dram_entity : elem->hbm_entity;
    auto reserved_mem = mem_type == HOST_SIDE ? elem->dram_gva : elem->hbm_gva;
    if (entity == nullptr) {
        SHM_LOG_DEBUG("entity not create, skip release.");
        return ACLSHMEM_SUCCESS;
    }

    // release reserved_mem
    if (reserved_mem != nullptr) {
        auto ret = hybm_unreserve_mem_space(entity, 0, reserved_mem);
        if (ret != 0) {
            SHM_LOG_INFO("unreserve mem space failed: " << ret);
        }
    } else {
        SHM_LOG_DEBUG("reserved_mem not reserved, skip release.");
    }

    // release entity resource
    hybm_destroy_entity(entity, 0);
    if (mem_type == HOST_SIDE) {
        elem->dram_slice = nullptr;
        elem->dram_gva = nullptr;
        elem->dram_entity = nullptr;
    }
    else {
        elem->hbm_slice = nullptr;
        elem->hbm_gva = nullptr;
        elem->hbm_entity = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_backend::aclshmemi_control_barrier_all()
{
    return g_boot_handle.barrier(&g_boot_handle);
}

int aclshmemi_init_backend::is_alloc_size_symmetric(size_t size)
{
    // fetch entity_member
    entity_member *elem = nullptr;
    {
        std::lock_guard<std::mutex> lock(entity_map_mutex_);
        auto iter = entity_map_.find(g_instance_ctx->id);
        if (iter == entity_map_.end() || iter->second == nullptr) {
            SHM_LOG_ERROR("Inner backend find instance context failed !");
            return ACLSHMEM_INNER_ERROR;
        }
        elem = iter->second;
    }

    auto host_state = elem->entity_host_state;
    auto boot_handle = elem->entity_boot_handle;
    if (host_state == nullptr || boot_handle == nullptr) {
        SHM_LOG_ERROR("One of entity_member's required fields is null: "
                      "host_state, boot_handle. Please Check!");
        return ACLSHMEM_INNER_ERROR;
    }

    std::vector<size_t> all_size(host_state->npes, 0);

    int ret = 0;
    ret = boot_handle->allgather(&size, all_size.data(), static_cast<int>(sizeof(size_t)), boot_handle);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("bootstrap allgather failed, ret: " <<ret);
        return ret;
    }

    for (int i = 0; i < host_state->npes; ++i) {
        SHM_LOG_INFO("alloc_size[pe " << i << "]=" << all_size[i]);
    }

    size_t ref = all_size[0];

    auto it = std::find_if(all_size.begin() + 1, all_size.end(), [ref](size_t alloc_size) {
        return alloc_size != ref;
    });

    if (it != all_size.end()) {
        int bad_pe = std::distance(all_size.begin(), it);
        size_t bad_val = *it;

        SHM_LOG_ERROR("Asymmetric alloc size detected, ref(pe0) = " << ref << ", first detected bad(pe" << bad_pe << ")= "  
                                        << bad_val << ", cur(pe" << host_state->mype << ")= " << all_size[host_state->mype]);
        return ACLSHMEM_INVALID_PARAM;
    }
    return ACLSHMEM_SUCCESS;
}