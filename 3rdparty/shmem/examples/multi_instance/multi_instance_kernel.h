/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MULTI_INSTANCE_KERNEL_H
#define MULTI_INSTANCE_KERNEL_H

#include <cstdint>

template <class T>
void group_allgather(
        uint32_t block_dim, void *stream, uint64_t fftsAddr, 
        uint8_t *input, uint8_t *output, uint8_t *gva, int elements, int magic, uint64_t instance_id);

#endif // MULTI_INSTANCE_KERNEL_H