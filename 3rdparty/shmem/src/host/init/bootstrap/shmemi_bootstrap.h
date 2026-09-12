/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEMI_BOOTSTRAP_H
#define SHMEMI_BOOTSTRAP_H
#include "shmem.h"
#ifdef __cplusplus
extern "C" {
#endif

int32_t aclshmemi_bootstrap_plugin_pre_init(aclshmemi_bootstrap_handle_t *handle);

int32_t aclshmemi_bootstrap_pre_init(int flags, aclshmemi_bootstrap_handle_t *handle);

int32_t aclshmemi_bootstrap_init(int flags, aclshmemx_init_attr_t *attr);

void aclshmemi_bootstrap_finalize();

int aclshmemi_bootstrap_plugin_init(void *mpi_comm, aclshmemi_bootstrap_handle_t *handle);

#ifdef __cplusplus
}
#endif
#endif