/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBM_COMMON_INCLUDE_H
#define MEM_FABRIC_HYBRID_HYBM_COMMON_INCLUDE_H

#include <map>
#include <mutex>

#include "mem_entity_def.h"

int32_t HybmGetInitDeviceId();

bool HybmHasInited();

#endif // MEM_FABRIC_HYBRID_HYBM_COMMON_INCLUDE_H
