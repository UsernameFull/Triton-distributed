/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UT_FUNC_TYPE_H
#define UT_FUNC_TYPE_H

#define ACLSHMEM_FUNC_TYPE_HOST(FUNC) \
    FUNC(half, op::fp16_t);           \
    FUNC(float, float);               \
    FUNC(double, double);             \
    FUNC(int8, int8_t);               \
    FUNC(int16, int16_t);             \
    FUNC(int32, int32_t);             \
    FUNC(int64, int64_t);             \
    FUNC(uint8, uint8_t);             \
    FUNC(uint16, uint16_t);           \
    FUNC(uint32, uint32_t);           \
    FUNC(uint64, uint64_t);           \
    FUNC(char, char);                 \
    FUNC(bfloat16, op::bfloat16)

#define ACLSHMEM_FUNC_TYPE_KERNEL(FUNC) \
    FUNC(half, half);                   \
    FUNC(float, float);                 \
    FUNC(double, double);               \
    FUNC(int8, int8_t);                 \
    FUNC(int16, int16_t);               \
    FUNC(int32, int32_t);               \
    FUNC(int64, int64_t);               \
    FUNC(uint8, uint8_t);               \
    FUNC(uint16, uint16_t);             \
    FUNC(uint32, uint32_t);             \
    FUNC(uint64, uint64_t);             \
    FUNC(char, char);                   \
    FUNC(bfloat16, bfloat16_t)

#define ACLSHMEM_MEM_PUT_GET_FUNC(FUNC) \
    FUNC(float, float);                 \
    FUNC(double, double);               \
    FUNC(int8, int8_t);                 \
    FUNC(int16, int16_t);               \
    FUNC(int32, int32_t);               \
    FUNC(int64, int64_t);               \
    FUNC(uint8, uint8_t);               \
    FUNC(uint16, uint16_t);             \
    FUNC(uint32, uint32_t);             \
    FUNC(uint64, uint64_t);             \
    FUNC(char, char)

#define ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_HOST(FUNC) \
    FUNC(bfloat16, op::bfloat16);                \
    FUNC(half, op::fp16_t);                      \
    FUNC(float, float);                          \
    FUNC(int8, int8_t);                          \
    FUNC(int16, int16_t);                        \
    FUNC(int32, int32_t)

#define ACLSHMEM_ATOMIC_ADD_FUNC_TYPE_KERNEL(FUNC) \
    FUNC(bfloat16, bfloat16_t);                    \
    FUNC(half, half);                              \
    FUNC(float, float);                            \
    FUNC(int8, int8_t);                            \
    FUNC(int16, int16_t);                          \
    FUNC(int32, int32_t)

/*****************************************************************************
 *                    RDMA Atomic Operation Type Macros                        *
 *****************************************************************************/
/* RDMA atomic operations - template based, supports all types */
#define ACLSHMEM_RDMA_ATOMIC_SWAP_FUNC_TYPE(FUNC) \
    FUNC(uint32, uint32_t);                       \
    FUNC(uint64, uint64_t);                       \
    FUNC(int32, int32_t);                         \
    FUNC(int64, int64_t)

#define ACLSHMEM_RDMA_ATOMIC_CAS_FUNC_TYPE(FUNC) \
    FUNC(uint32, uint32_t);                      \
    FUNC(uint64, uint64_t);                      \
    FUNC(int32, int32_t);                        \
    FUNC(int64, int64_t)

#define ACLSHMEM_RDMA_ATOMIC_ADD_FUNC_TYPE(FUNC) \
    FUNC(uint32, uint32_t);                      \
    FUNC(uint64, uint64_t);                      \
    FUNC(int32, int32_t);                        \
    FUNC(int64, int64_t)

#define ACLSHMEM_RDMA_ATOMIC_LOGIC_FUNC_TYPE(FUNC) \
    FUNC(uint32, uint32_t);                        \
    FUNC(uint64, uint64_t);                        \
    FUNC(int32, int32_t);                          \
    FUNC(int64, int64_t)

/*****************************************************************************
 *                  UDMA Atomic Operation Type Macros                         *
 *****************************************************************************/

#define UDMA_ATOMIC_ADD_FUNC_TYPE(FUNC) \
    FUNC(int32, int32_t);               \
    FUNC(uint32, uint32_t);             \
    FUNC(int64, int64_t);               \
    FUNC(uint64, uint64_t);             \
    FUNC(float, float)

#define UDMA_ATOMIC_FUNC_TYPE(FUNC) \
    FUNC(int32, int32_t);           \
    FUNC(uint32, uint32_t);         \
    FUNC(int64, int64_t);           \
    FUNC(uint64, uint64_t)

struct uint128_t {
    uint64_t x;
    uint64_t y;
    uint128_t(uint64_t x, uint64_t y) : x(x), y(y) {}
    uint128_t(uint64_t x) : x(x), y(0) {}
    uint128_t& operator=(uint64_t value)
    {
        x = value;
        y = 0;
        return *this;
    }
};

constexpr bool operator==(const uint128_t& lhs, const uint128_t& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }

constexpr bool operator!=(const uint128_t& lhs, const uint128_t& rhs) { return !(lhs == rhs); }

#define ACLSHMEM_PUT_SIZE_FUNC(FUNC) \
    FUNC(8, uint8_t);                \
    FUNC(16, uint16_t);              \
    FUNC(32, uint32_t);              \
    FUNC(64, uint64_t);              \
    FUNC(128, uint128_t)

#define ACLSHMEM_PUT_GET_SIZE_FUNC ACLSHMEM_PUT_SIZE_FUNC

#endif // UT_FUNC_TYPE_H
