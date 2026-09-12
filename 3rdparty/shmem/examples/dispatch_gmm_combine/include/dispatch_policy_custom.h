/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DISPATCH_POLICY_CUSTOM_H
#define DISPATCH_POLICY_CUSTOM_H
namespace Catlass::Gemm {
    template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
    struct MmadAtlasA2PreloadFixpipeQuant : public MmadAtlasA2 {
        static constexpr uint32_t STAGES = 2;
        static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
        static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
    };

    template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
        uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
    struct MmadAtlasA2PreloadAsyncFixpipe :
        public MmadAtlasA2PreloadAsync<
            PRELOAD_STAGES_,
            L1_STAGES_,
            L0A_STAGES_,
            L0B_STAGES_,
            L0C_STAGES_,
            ENABLE_UNIT_FLAG_,
            ENABLE_SHUFFLE_K_
        > {
    };
}

namespace Catlass::Epilogue {
    template <uint32_t UB_STAGES_>
    struct EpilogueAtlasA2UnQuant {
        using ArchTag = Arch::AtlasA2;
        static constexpr uint32_t UB_STAGES = UB_STAGES_;
    };

    template <uint32_t UB_STAGES_>
    struct EpilogueAtlasA2PerTokenDequantSwigluQuant {
        using ArchTag = Arch::AtlasA2;
        static constexpr uint32_t UB_STAGES = UB_STAGES_;
    };
}
#endif