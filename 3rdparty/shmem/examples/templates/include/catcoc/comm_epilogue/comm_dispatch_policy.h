/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCOC_EPILOGUE_DISPATCH_POLICY_H
#define CATCOC_EPILOGUE_DISPATCH_POLICY_H

// from catlass
#include "catlass/arch/arch.hpp"

#include "catcoc/detail/remote_copy_type.h"

namespace Catcoc::CommEpilogue {

// For AtlasA2, an remote copy epilogue of the form D(share mem) = C(share mem)
template <uint32_t UB_STAGES_, detail::CopyMode CopyMode_, bool IsDynamic_ = false>
struct EpilogueAtlasA2CommToShareMem {
    using ArchTag = Catlass::Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IsDynamic = IsDynamic_;
};

// For AtlasA2, an remote copy epilogue of the form D(local mem) = C(share mem)
template <uint32_t UB_STAGES_, detail::CopyMode CopyMode_, bool IsDynamic_ = false>
struct EpilogueAtlasA2CommToLocalMem {
    using ArchTag = Catlass::Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IsDynamic = IsDynamic_;
};

template <uint32_t UB_STAGES_, detail::CopyMode CopyMode_, bool IsDynamic_ = false>
struct EpilogueAtlasA2CommRemoteCopy {
    using ArchTag = Catlass::Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IsDynamic = IsDynamic_;
};

template <uint32_t UB_STAGES_, bool IsDynamic_ = false>
struct EpilogueAtlasA2CommLocalCopy {
    using ArchTag = Catlass::Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES;
    static constexpr bool IsDynamic = IsDynamic_;
};

}  // namespace Catcoc::CommEpilogue

#endif  // CATCOC_EPILOGUE_DISPATCH_POLICY_H
