/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_MM_H
#define SHMEMI_MM_H

#include <pthread.h>
#include <cstdint>
#include <map>
#include <set>

#include "host/shmem_host_def.h"

int32_t memory_manager_initialize(void *base, uint64_t size, aclshmem_mem_type_t mem_type = DEVICE_SIDE);
void memory_manager_destroy();

#endif  // ACLSHMEMI_MM_H
