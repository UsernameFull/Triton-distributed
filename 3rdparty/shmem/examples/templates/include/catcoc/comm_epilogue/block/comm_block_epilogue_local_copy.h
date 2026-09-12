/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCOC_COMM_EPILOGUE_BLOCK_EPILOGUE_BLOCK_LOCAL_COPY_H
#define CATCOC_COMM_EPILOGUE_BLOCK_EPILOGUE_BLOCK_LOCAL_COPY_H

#include "catcoc/catcoc.h"
#include "catcoc/comm_epilogue/comm_dispatch_policy.h"
#include "catcoc/detail/remote_copy_type.h"

// from catlass
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

namespace Catcoc::CommEpilogue::Block {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    uint32_t UB_STAGES_,
    bool IsDynamic_,
    class SrcType_,
    class DstType_,
    class BlockShape_,
    class TileShape_,
    class EpilogueTileSwizzle_
>
class CommBlockEpilogue <
    EpilogueAtlasA2CommLocalCopy<UB_STAGES_, IsDynamic_>,
    SrcType_,
    DstType_,
    BlockShape_,
    TileShape_,
    EpilogueTileSwizzle_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2CommLocalCopy<UB_STAGES_, IsDynamic_>;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IsDynamic = IsDynamic_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;

    using BlockShape = BlockShape_;
    using TileShape = TileShape_;

    using CopyGmToUb = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, SrcType_>;
    using CopyUbToGm = Catlass::Epilogue::Tile::CopyUb2Gm<ArchTag, DstType_>;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    // Epilogue params definition
    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_DEVICE
        static MatrixCoord BlockShape() { return BlockShape::ToCoord(); }
        CATLASS_DEVICE
        static MatrixCoord TileShape() { return TileShape::ToCoord(); }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord blockShape;
        MatrixCoord tileShape;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(MatrixCoord blockShape_, MatrixCoord tileShape_) : blockShape(blockShape_), tileShape(tileShape_) {}

        CATLASS_DEVICE
        MatrixCoord BlockShape() const { return blockShape; }
        CATLASS_DEVICE
        MatrixCoord TileShape() const { return tileShape; }
    };

    using Params = ParamsBase<IsDynamic>;

    CATLASS_DEVICE
    CommBlockEpilogue(Catlass::Arch::Resource<ArchTag> &resource, Params const &params) : params(params)
    {
        size_t ubOffset = 0;
        uint32_t eventUbMte3Mte2Id = 0;
        uint32_t eventUbMte2Mte3Id = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubList[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(ubOffset);
            ubOffset += params.TileShape().row() * params.TileShape().column() * sizeof(ElementSrc);

            eventUbMte3Mte2List[i] = eventUbMte3Mte2Id++;
            eventUbMte2Mte3List[i] = eventUbMte2Mte3Id++;
        }
    }

    CATLASS_DEVICE
    void InitBlockLoop()
    {
        ubListId = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbMte3Mte2List[i]);
        }
    }

    CATLASS_DEVICE
    void FinalizeBlockLoop()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbMte3Mte2List[i]);
        }
    }

    CATLASS_DEVICE
    ~CommBlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const &layoutSrc,
        AscendC::GlobalTensor<ElementDst> const& gmDst, LayoutDst const &layoutDst,
        MatrixCoord const &actualBlockShape
    )
    {
        if (actualBlockShape.row() == 0) {
            return;
        }

        auto tileShape = params.TileShape();
        auto ubTileStride = Catlass::MakeCoord<int64_t>(tileShape.column(), 1);
        EpilogueTileSwizzle epilogueTileSwizzle{actualBlockShape, tileShape};
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();

        for (uint32_t tileIdx = 0; tileIdx < tileLoops; tileIdx++) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(tileIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;

            auto &ubTile = ubList[ubListId];
            LayoutSrc layoutUb{actualTileShape, ubTileStride};

            // Get the data and layout of input
            auto gmTileSrc = gmSrc[layoutSrc.GetOffset(tileOffsetInBlock)];
            auto layoutTileSrc = layoutSrc.GetTileLayout(actualTileShape);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbMte3Mte2List[ubListId]);
            copyGmToUb(ubTile, gmTileSrc, layoutUb, layoutTileSrc);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventUbMte2Mte3List[ubListId]);

            // Get the data and layout of output
            auto gmTileDst = gmDst[layoutDst.GetOffset(tileOffsetInBlock)];
            auto layoutTileDst = layoutDst.GetTileLayout(actualTileShape);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventUbMte2Mte3List[ubListId]);
            copyUbToGm(gmTileDst, ubTile, layoutTileDst, layoutUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbMte3Mte2List[ubListId]);
            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementSrc> ubList[UB_STAGES];
    uint32_t eventUbMte3Mte2List[UB_STAGES];
    uint32_t eventUbMte2Mte3List[UB_STAGES];
    uint32_t ubListId{0};
    CopyGmToUb copyGmToUb;
    CopyUbToGm copyUbToGm;
};

} // namespace Catcoc::CommEpilogue::Block

#endif // CATCOC_COMM_EPILOGUE_BLOCK_EPILOGUE_BLOCK_LOCAL_COPY_H