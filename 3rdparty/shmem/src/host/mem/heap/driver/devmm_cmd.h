/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HYBM_DEVMM_CMD_H
#define HYBM_DEVMM_CMD_H

#include <string>

namespace shm {
namespace drv {

void DevmmInitialize(int deviceId, int fd) noexcept;

int DevmmMapShareMemory(const char *name, void *expectAddr, uint64_t size, uint64_t flags) noexcept;

int DevmmUnmapShareMemory(void *expectAddr, uint64_t flags) noexcept;

int DevmmIoctlAllocAnddAdvice(uint64_t ptr, size_t size, uint32_t devid, uint32_t advise) noexcept;

}
}

#endif  // HYBM_DEVMM_CMD_H
