/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_SIMT_COMMON_H
#define SHMEMI_DEVICE_SIMT_COMMON_H

#include "shmemi_def.h"
#include "shmemi_device_simt_common.hpp"

namespace simt
{

__simt_callee__ inline __gm__ aclshmem_device_host_state_simt_t *aclshmemi_get_state();

__simt_callee__ inline int aclshmemi_get_my_pe();

__simt_callee__ inline int aclshmemi_get_total_pe();

__simt_callee__ inline uint64_t aclshmemi_get_heap_size();

template<typename T>
__simt_callee__ inline void aclshmemi_store(__gm__ T *addr, T val);

template<typename T>
__simt_callee__ inline T aclshmemi_load(__gm__ T *cache);

} // namespace simt

#endif