/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBM_DEFINE_H
#define MEM_FABRIC_HYBRID_HYBM_DEFINE_H

#include <netinet/in.h>
#include <cstdint>
#include <cstddef>

namespace shm {

constexpr uint64_t DEVICE_LARGE_PAGE_SIZE = 2UL * 1024UL * 1024UL;  // 大页的size, 2M
constexpr uint64_t HYBM_DEVICE_VA_START = 0x100000000000UL;         // NPU上的地址空间起始: 16T
constexpr uint64_t HYBM_DEVICE_VA_SIZE = 0x80000000000UL;           // NPU上的地址空间范围: 8T
constexpr uint64_t HYBM_DEVICE_VA_END = HYBM_DEVICE_VA_START + HYBM_DEVICE_VA_SIZE;
constexpr uint64_t SVM_END_ADDR = HYBM_DEVICE_VA_START + HYBM_DEVICE_VA_SIZE - (1UL << 30UL); // svm的结尾虚拟地址
constexpr uint64_t HYBM_DEVICE_PRE_META_SIZE = 128UL; // 128B
constexpr uint64_t HYBM_DEVICE_GLOBAL_META_SIZE = HYBM_DEVICE_PRE_META_SIZE; // 128B
constexpr uint64_t HYBM_ENTITY_NUM_MAX = 511UL; // entity最大数量
constexpr uint64_t HYBM_DEVICE_META_SIZE = HYBM_DEVICE_PRE_META_SIZE * HYBM_ENTITY_NUM_MAX
    + HYBM_DEVICE_GLOBAL_META_SIZE; // 64K

constexpr uint64_t HYBM_DEVICE_USER_CONTEXT_PRE_SIZE = 64UL * 1024UL; // 64K
constexpr uint64_t HYBM_DEVICE_INFO_SIZE = HYBM_DEVICE_USER_CONTEXT_PRE_SIZE * HYBM_ENTITY_NUM_MAX
    + HYBM_DEVICE_META_SIZE; // 元数据+用户context,总大小32M, 对齐DEVICE_LARGE_PAGE_SIZE
constexpr uint64_t HYBM_DEVICE_META_ADDR = SVM_END_ADDR - HYBM_DEVICE_INFO_SIZE;
constexpr uint64_t HYBM_DEVICE_USER_CONTEXT_ADDR = HYBM_DEVICE_META_ADDR + HYBM_DEVICE_META_SIZE;
constexpr uint32_t ACL_MEMCPY_HOST_TO_HOST = 0;
constexpr uint32_t ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr uint32_t ACL_MEMCPY_DEVICE_TO_HOST = 2;
constexpr uint32_t ACL_MEMCPY_DEVICE_TO_DEVICE = 3;

constexpr uint64_t DEVMM_HEAP_SIZE = (1UL << 30UL);     // same definition in devmm

constexpr uint64_t HYBM_HOST_CONN_START_ADDR = 0x30000000000UL;  // 48T
constexpr uint64_t HYBM_HOST_CONN_ADDR_SIZE = 0x100000000000UL;  // 16T

constexpr uint64_t HYBM_GVM_START_ADDR = 0x280000000000UL;      // 40T
constexpr uint64_t HYBM_GVM_END_ADDR = 0xA80000000000UL;        // 168T

inline bool IsVirtualAddressNpu(uint64_t address)
{
    return (address >= HYBM_DEVICE_VA_START && address < (HYBM_DEVICE_VA_START + HYBM_DEVICE_VA_SIZE));
}

inline bool IsVirtualAddressNpu(const void *address)
{
    return IsVirtualAddressNpu(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)));
}

inline uint64_t Valid48BitsAddress(uint64_t address)
{
    return address & 0xffffffffffffUL;
}

inline const void *Valid48BitsAddress(const void *address)
{
    uint64_t addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address));
    return reinterpret_cast<const void *>(static_cast<uintptr_t>(Valid48BitsAddress(addr)));
}

inline void *Valid48BitsAddress(void *address)
{
    uint64_t addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address));
    return reinterpret_cast<void *>(static_cast<uintptr_t>(Valid48BitsAddress(addr)));
}

struct HybmDeviceMeta {
    uint32_t entityId;
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t extraContextSize;
    uint64_t symmetricSize;
    uint64_t qpInfoAddress;
    uint64_t reserved[12]; // total 128B, equal HYBM_DEVICE_PRE_META_SIZE
};
}  // namespace shm

#endif  // MEM_FABRIC_HYBRID_HYBM_DEFINE_H
