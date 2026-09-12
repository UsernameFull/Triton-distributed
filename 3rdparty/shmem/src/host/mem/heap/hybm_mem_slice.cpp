/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>

#include "hybm_mem_slice.h"

namespace shm {
union SliceIdUnion {
    struct SliceDetail {
        uint64_t magic_ : 40;         /* to verify hybm_mem_slice_t ptr */
        uint64_t index_ : 16;         /* id of mem slice  */
        uint64_t memType_ : 4;        /* device or host memory */
        uint64_t memPageTblType_ : 2; /* use CANN SVM page table or HyBM page table */
        uint64_t reserved : 2;
    } detail;
    uint64_t number;
    void *address;
};

hybm_mem_slice_t MemSlice::ConvertToId() const noexcept
{
    SliceIdUnion idUnion{};
    idUnion.detail.magic_ = magic_;
    idUnion.detail.index_ = index_;
    idUnion.detail.memType_ = memType_;
    idUnion.detail.memPageTblType_ = memPageTblType_;
    idUnion.detail.reserved = 0;
    return idUnion.address;
}

bool MemSlice::ValidateId(hybm_mem_slice_t slice) const
{
    SliceIdUnion idUnion{};
    idUnion.address = slice;
    return idUnion.address == ConvertToId();
}

uint64_t MemSlice::GetIndexFrom(hybm_mem_slice_t slice) noexcept
{
    SliceIdUnion idUnion{};
    idUnion.address = slice;
    return idUnion.detail.index_;
}
}