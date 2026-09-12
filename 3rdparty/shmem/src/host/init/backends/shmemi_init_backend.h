/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_INIT_BACKEND_H
#define SHMEMI_INIT_BACKEND_H

#include <map>
#include <mutex>

#include "host/shmem_host_def.h"
#include "host_device/shmem_common_types.h"

#include "mem/shmemi_mgr.h"

#include "init/bootstrap/shmemi_bootstrap.h"

#include "hybm_def.h"

#include "utils/mstx/mstx_mem_register.h"

typedef struct entity_member {
    void *hbm_gva = nullptr;
    void *dram_gva = nullptr;

    hybm_entity_t hbm_entity = nullptr;
    hybm_entity_t dram_entity = nullptr;
    hybm_mem_slice_t hbm_slice = nullptr;
    hybm_mem_slice_t dram_slice = nullptr;

    aclshmemx_init_attr_t *entity_attr;
    aclshmemi_bootstrap_handle_t *entity_boot_handle;
    aclshmem_device_host_state_t *entity_host_state;
    aclshmem_device_host_state_t *entity_device_state;
} entity_member;


class aclshmemi_init_backend {
public:
    aclshmemi_init_backend();
    ~aclshmemi_init_backend();

    int init_device_state();
    int finalize_device_state();
    int update_device_state(void* host_ptr, size_t size);

    int reserve_heap(aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    int setup_heap(aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    int remove_heap(aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    int release_heap(aclshmem_mem_type_t mem_type = DEVICE_SIDE);

    int aclshmemi_control_barrier_all();
    int is_alloc_size_symmetric(size_t size);

    int bind_aclshmem_entity(aclshmemx_init_attr_t *attr, aclshmem_device_host_state_t *state, aclshmemi_bootstrap_handle_t *handle);
    int release_aclshmem_entity(uint64_t instance_id);

private:
    int reach_info_init(void *&gva);
    int create_entity(aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    int exchange_slice(aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    int exchange_entity(aclshmem_mem_type_t mem_type = DEVICE_SIDE);

private:
    int device_id;
    int hybm_init_count = 0;

    shm::mstx_mem_register_base* mstx_reg_ptr_ = nullptr;

    std::mutex entity_map_mutex_;
    std::map<uint64_t, entity_member*> entity_map_ = {};
};

#endif // SHMEMI_INIT_BACKEND_H
