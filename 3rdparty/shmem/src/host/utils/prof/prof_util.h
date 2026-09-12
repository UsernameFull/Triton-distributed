/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROF_UTIL_H
#define PROF_UTIL_H

#include "host_device/shmem_common_types.h"

int32_t prof_get_col_pe(char *tmp_pe);

int32_t prof_util_init(aclshmem_prof_pe_t *host_profs, aclshmem_device_host_state_t *global_state);

void prof_data_print(aclshmem_prof_pe_t *host_profs, aclshmem_device_host_state_t *global_state, 
                     aclshmem_prof_pe_t **out_profs, bool verbose = true);

#endif  // PROF_UTIL_H