/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef COPY_L0C_TO_GM_CUSTOM_H
#define COPY_L0C_TO_GM_CUSTOM_H
namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct ShmCopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};
// Fixpipe with quant VDEQF16
template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct ShmCopyL0CToGm<Catlass::Arch::AtlasA2,
                ElementAccumulator_,
                Gemm::GemmType<ElementDst_, layout::RowMajor>,
                ScaleGranularity::PER_CHANNEL,
                ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
                    AscendC::LocalTensor<uint64_t> cbufWorkspace,
                    LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;
        constexpr uint32_t DST_LAYOUT_N_INDEX = 1;
        constexpr uint32_t DST_LAYOUT_M_INDEX = 0;
        constexpr uint32_t LAYOUT_STRIDE_3_INDEX = 3;
        constexpr uint32_t LAYOUT_STRIDE_0_INDEX = 0;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(DST_LAYOUT_N_INDEX);
        intriParams.mSize = dstLayout.shape(DST_LAYOUT_M_INDEX);
        intriParams.srcStride = srcLayout.stride(LAYOUT_STRIDE_3_INDEX) / srcLayout.stride(LAYOUT_STRIDE_0_INDEX);
        intriParams.dstStride = dstLayout.stride(LAYOUT_STRIDE_0_INDEX);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, cbufWorkspace, intriParams);
    }
};
}
#endif