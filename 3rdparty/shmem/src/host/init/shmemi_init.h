/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_INIT_H
#define SHMEMI_INIT_H

#include <memory>
#include "stdint.h"
#include "host_device/shmem_common_types.h"
#include "init/backends/shmemi_init_backend.h"

extern aclshmem_device_host_state_t g_state;
extern aclshmem_host_state_t g_state_host;
extern aclshmem_prof_pe_t g_host_profs;

// bootstrap plugin_hdl
extern void *plugin_hdl;
extern aclshmemi_bootstrap_handle_t g_boot_handle;
extern std::shared_ptr<memory_manager> aclshmemi_memory_manager;
extern std::shared_ptr<memory_manager> aclshmemi_host_memory_manager;

extern aclshmemi_init_backend* init_manager;
extern aclshmem_instance_ctx *g_instance_ctx;

int32_t aclshmemi_control_barrier_all();
int32_t is_alloc_size_symmetric(size_t size);

int32_t update_device_state(void);
int32_t aclshmemx_instance_ctx_set_impl(uint64_t instance_id);

#endif  // ACLSHMEMI_INIT_H
