/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_DL_RT_DEF_H
#define MF_HYBRID_DL_RT_DEF_H

#include <cstdint>

namespace shm {
typedef enum {
    RT_STREAM_CREATE_ATTR_FLAGS = 1,        // Stream创建flags
    RT_STREAM_CREATE_ATTR_PRIORITY = 2,     // Stream优先级
    RT_STREAM_CREATE_ATTR_MAX = 3
} rtStreamCreateAttrId;

typedef union {
    uint32_t flags;
    uint32_t priority;
    uint32_t rsv[4];
} rtStreamCreateAttrValue_t;

typedef struct {
    rtStreamCreateAttrId id;    // 属性id
    rtStreamCreateAttrValue_t value; // 属性值
} rtStreamCreateAttr_t;

typedef struct {
    rtStreamCreateAttr_t *attrs; // attrs配置的值
    size_t numAttrs;  // attrs数量
} rtStreamCreateConfig_t;
} // namespace shm

#endif  // MF_HYBRID_DL_RT_DEF_H
