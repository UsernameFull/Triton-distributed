/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __MEMFABRIC_SHMEM_SIMT_AI_CORE_BASE_META_H__
#define __MEMFABRIC_SHMEM_SIMT_AI_CORE_BASE_META_H__

namespace simt
{

__simt_callee__ inline uint32_t aclshmemi_get_global_rank(uint32_t shmemId)
{
    if (shmemId >= SMEM_OBJECT_NUM_MAX) {
        return UINT32_MAX;
    }
    uint64_t metaAddr = SMEM_SHM_DEVICE_META_ADDR + SMEM_SHM_DEVICE_GLOBAL_META_SIZE +
        shmemId * SMEM_SHM_DEVICE_PRE_META_SIZE;
    return (*(__gm__ uint32_t*)(metaAddr + SMEM_SHM_DEVICE_META_RANK_OFFSET));
}

__simt_callee__ inline uint32_t aclshmemi_get_global_rank_size(uint32_t shmemId)
{
    if (shmemId >= SMEM_OBJECT_NUM_MAX) {
        return UINT32_MAX;
    }
    uint64_t metaAddr = SMEM_SHM_DEVICE_META_ADDR + SMEM_SHM_DEVICE_GLOBAL_META_SIZE +
        shmemId * SMEM_SHM_DEVICE_PRE_META_SIZE;
    return (*(__gm__ uint32_t*)(metaAddr + SMEM_SHM_DEVICE_META_RANK_SIZE_OFFSET));
}

__simt_callee__ inline uint64_t aclshmemi_get_symmetric_size(uint32_t shmemId)
{
    if (shmemId >= SMEM_OBJECT_NUM_MAX) {
        return 0;
    }
    uint64_t metaAddr = SMEM_SHM_DEVICE_META_ADDR + SMEM_SHM_DEVICE_GLOBAL_META_SIZE +
        shmemId * SMEM_SHM_DEVICE_PRE_META_SIZE;
    return (*(__gm__ uint64_t*)(metaAddr + SMEM_SHM_DEVICE_META_SYMM_OFFSET));
}

__simt_callee__ inline __gm__ void* aclshmemi_get_qp_info_address(uint32_t shmemId)
{
    if (shmemId >= SMEM_OBJECT_NUM_MAX) {
        return NULL;
    }

    uint64_t metaAddr = SMEM_SHM_DEVICE_META_ADDR + SMEM_SHM_DEVICE_GLOBAL_META_SIZE +
        shmemId * SMEM_SHM_DEVICE_PRE_META_SIZE;
    return *(__gm__ void **)(metaAddr + SMEM_SHM_DEVICE_META_QP_INFO_OFFSET);
}

__simt_callee__ inline __gm__ void* aclshmemi_get_extra_context_addr(uint32_t shmemId)
{
    if (shmemId >= SMEM_OBJECT_NUM_MAX) {
        return NULL;
    }
    uint64_t ctxAddr = SMEM_SHM_DEVICE_USER_CONTEXT_ADDR + shmemId * SMEM_SHM_DEVICE_USER_CONTEXT_PRE_SIZE;
    return ((__gm__ void*)(ctxAddr));
}

__simt_callee__ inline uint32_t aclshmemi_get_extra_context_size(uint32_t shmemId)
{
    if (shmemId >= SMEM_OBJECT_NUM_MAX) {
        return 0;
    }
    uint64_t metaAddr = SMEM_SHM_DEVICE_META_ADDR + SMEM_SHM_DEVICE_GLOBAL_META_SIZE +
        shmemId * SMEM_SHM_DEVICE_PRE_META_SIZE;
    return (*(__gm__ uint32_t*)(metaAddr + SMEM_SHM_DEVICE_META_CONTEXT_OFFSET));
}

} // namespace simt
#endif // __MEMFABRIC_SMEM_AI_CORE_BASE_META_H__