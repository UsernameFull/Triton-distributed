/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEM_TORCH_KERNEL_H
#define ACLSHMEM_TORCH_KERNEL_H

#include <acl/acl.h>
#include <vector>

namespace ShmemKernel {

int aclshmem_kv_shuffle(uint32_t block_dim, aclrtStream stream, uint64_t fftsAddr, void* k_cache,
    void* v_cache,
    void* global_shuffle_table,
    void* src_block_table,
    void* dst_block_table,
    void* sync_ptr,
    int64_t block_nums, int64_t kv_head_num, int64_t page_size, int64_t head_dim, int32_t sync_count);

template <class T>
void aclshmem_allgather(uint32_t block_dim, void *stream, uint64_t fftsAddr, void *input, void *output, void *gva,
                    int elements, int magic);

void aclshmem_allgather_matmul(uint32_t block_dim, void *stream, uint64_t fftsAddr, void *aDevice, void *bDevice, void *cDevice, void *gmSymmetric, uint32_t m, uint32_t n, uint32_t k);

}

#endif // ACLSHMEM_TORCH_KERNEL_H