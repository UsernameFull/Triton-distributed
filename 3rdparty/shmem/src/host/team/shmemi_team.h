/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_TEAM_H
#define SHMEMI_TEAM_H

#include "stdint.h"
#include "host_device/shmem_common_types.h"

int32_t aclshmemi_team_init(int32_t rank, int32_t size);

int32_t aclshmemi_team_finalize();

void aclshmemi_team_populate_from_world_pe_mapping(aclshmemx_team_t *team);

void aclshmemi_team_populate_pe_mappings_from_constant_stride(aclshmemx_team_t *team);

int32_t aclshmemi_team_pe_mapping(aclshmem_team_t team, int pe);

#endif  // ACLSHMEMI_TEAM_H
