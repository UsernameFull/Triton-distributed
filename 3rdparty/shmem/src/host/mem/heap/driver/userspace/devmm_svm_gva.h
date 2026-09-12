/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEM_FABRIC_HYBRID_DEVMM_SVM_GVA_H
#define MEM_FABRIC_HYBRID_DEVMM_SVM_GVA_H

#include <cstdint>

namespace shm {
namespace drv {

const uint64_t GVA_GIANT_FLAG = (1ULL << 0);

int32_t HalGvaReserveMemory(uint64_t *address, size_t size, int32_t deviceId, uint64_t flags);

int32_t HalGvaUnreserveMemory(uint64_t address);

int32_t HalGvaAlloc(uint64_t address, size_t size, uint64_t flags);

int32_t HalGvaFree(uint64_t address, size_t size);

int32_t HalGvaOpen(uint64_t address, const char *name, size_t size, uint64_t flags);

int32_t HalGvaClose(uint64_t address, uint64_t flags);

}
}

#endif // MEM_FABRIC_HYBRID_DEVMM_SVM_GVA_H
