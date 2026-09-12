/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_UTILS_MSTX_REGISTER_BASE_H
#define SHMEM_UTILS_MSTX_REGISTER_BASE_H

#include <cstdint>
#include <vector>

constexpr uint32_t MSTX_GROUP_FOR_STATE = 0; 
constexpr uint32_t MSTX_GROUP_FOR_HOST_HEAP = 1; 
constexpr uint32_t MSTX_GROUP_FOR_DEVICE_HEAP = 2;

#ifdef __cplusplus
namespace shm {
#endif

class mstx_mem_register_base {
public:
    virtual ~mstx_mem_register_base() noexcept = default;
    virtual void mstx_mem_regions_register(int group = 0) = 0;
    virtual void mstx_mem_regions_unregister(int group = 0) = 0;
    virtual void add_mem_regions(void *ptr, uint64_t size, int group = 0) = 0;
    virtual void add_mem_regions_multi_pe_align(void *ptr, uint64_t size, uint64_t align_size, uint32_t pe_size, int group = 0) = 0;

    mstx_mem_register_base(const mstx_mem_register_base&) = delete;
    mstx_mem_register_base& operator=(const mstx_mem_register_base&) = delete;
    mstx_mem_register_base(mstx_mem_register_base&&) = delete;
    mstx_mem_register_base& operator=(mstx_mem_register_base&&) = delete;

protected:
    mstx_mem_register_base() = default;
};

#ifdef __cplusplus
extern "C" {
#endif
shm::mstx_mem_register_base* create_mstx_mem_register_instance(void);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
} /* namespace shm */
#endif

#endif /* SHMEM_UTILS_MSTX_REGISTER_BASE_H */