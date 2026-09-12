/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCOC_COMM_EPILOGUE_BLOCK_EPILOGUE_TO_LOCAL_MEM_H
#define CATCOC_COMM_EPILOGUE_BLOCK_EPILOGUE_TO_LOCAL_MEM_H

#include "catcoc/catcoc.h"
#include "catcoc/comm_epilogue/comm_dispatch_policy.h"
#include "catcoc/detail/remote_copy_type.h"

// from catlass
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

namespace Catcoc::CommEpilogue::Block {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    uint32_t UB_STAGES_,
    detail::CopyMode CopyMode_,
    bool IsDynamic_,
    class SrcType_,
    class DstType_,
    class CoreSplit_,
    class BlockShape_,
    class TileShape_,
    class TileRemoteCopy_,
    class EpilogueTileSwizzle_,
    class GemmRemapper_
>
class CommBlockEpilogue <
    EpilogueAtlasA2CommToLocalMem<UB_STAGES_, CopyMode_, IsDynamic_>,
    SrcType_,
    DstType_,
    CoreSplit_,
    BlockShape_,
    TileShape_,
    TileRemoteCopy_,
    EpilogueTileSwizzle_,
    GemmRemapper_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2CommToLocalMem<UB_STAGES_, CopyMode_, IsDynamic_>;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IsDynamic = IsDynamic_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;

    using CoreSplit = CoreSplit_;
    using BlockShape = BlockShape_;
    using TileShape = TileShape_;
    using TileRemoteCopy = TileRemoteCopy_;
    using EpilogueTileSwizzle = EpilogueTileSwizzle_;
    using GemmRemapper = GemmRemapper_;
    static constexpr detail::CopyMode RemoteCopyMode = CopyMode_;
    static constexpr detail::CopyDirect RemoteCopyDirect = TileRemoteCopy::RemoteCopyDirect;

    // Epilogue params definition
    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        __gm__ ElementDst *aclshmemPtr{nullptr};
        LayoutDst aclshmemLayout;
        GemmRemapper gemmRemapper;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(__gm__ ElementDst *aclshmemPtr_, LayoutDst const &aclshmemLayout_, GemmRemapper const &gemmRemapper_)
            : aclshmemPtr(aclshmemPtr_), aclshmemLayout(aclshmemLayout_), gemmRemapper(gemmRemapper_) {}

        CATLASS_DEVICE
        static MatrixCoord CoreSplit() { return CoreSplit::ToCoord(); }
        CATLASS_DEVICE
        static MatrixCoord BlockShape() { return BlockShape::ToCoord(); }
        CATLASS_DEVICE
        static MatrixCoord TileShape() { return TileShape::ToCoord(); }
    };

    template <>
    struct ParamsBase<true> {
        __gm__ ElementDst *aclshmemPtr{nullptr};
        LayoutDst aclshmemLayout;
        GemmRemapper gemmRemapper;
        MatrixCoord coreSplit;
        MatrixCoord blockShape;
        MatrixCoord tileShape;

        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(__gm__ ElementDst *aclshmemPtr_, LayoutDst const &aclshmemLayout_, GemmRemapper const &gemmRemapper_,
            MatrixCoord coreSplit_, MatrixCoord blockShape_, MatrixCoord tileShape_)
            : aclshmemPtr(aclshmemPtr_), aclshmemLayout(aclshmemLayout_), gemmRemapper(gemmRemapper_),
              coreSplit(coreSplit_), blockShape(blockShape_), tileShape(tileShape_) {}

        CATLASS_DEVICE
        MatrixCoord CoreSplit() const { return coreSplit; }
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

        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubSList[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(ubOffset);
            ubOffset += params.TileShape().row() * params.TileShape().column() * sizeof(ElementDst);
        }
    }

    CATLASS_DEVICE
    void AllocEventID()
    {
        uint32_t copyEventId = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            copyEventIdList[i] = copyEventId++;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[i]);
        }
    }

    CATLASS_DEVICE
    void ReleaseEventID()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[i]);
        }
        ubListId = 0;
    }

    CATLASS_DEVICE
    ~CommBlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void operator() (
        MatrixCoord const &gemmBlockShape,
        MatrixCoord const &outputBlockOffset,
        MatrixCoord const &inputBlockOffset,
        MatrixCoord const &commBlockShape,
        AscendC::GlobalTensor<ElementSrc> const& gmC,
        LayoutSrc const &layoutC,
        uint32_t const &globalLoopIdx,
        uint32_t const &rankIdx)
    {
        // Remap the idx & actual shape of the gemm block
        GemmCoord remapOutputBlockCoordMNK = params.gemmRemapper.GetBlockCoord(globalLoopIdx);
        GemmCoord actualGemmBlockShapeMNK = params.gemmRemapper.GetActualBlockShape(remapOutputBlockCoordMNK);
        MatrixCoord remapOutputBlockCoord = remapOutputBlockCoordMNK.GetCoordMN();
        MatrixCoord actualGemmBlockShape = actualGemmBlockShapeMNK.GetCoordMN();

        // Calculate the actual output offset of the communication block
        if (gemmBlockShape == 0) {
            return;
        }
        MatrixCoord blockOutterOffset = remapOutputBlockCoord * gemmBlockShape;
        MatrixCoord blockInnerOffset = outputBlockOffset % gemmBlockShape;

        // Get actual communication block shape
        MatrixCoord actualCommBlockShape;
        if (blockInnerOffset.row() < actualGemmBlockShape.row()) {
            actualCommBlockShape = MatrixCoord::Min(actualGemmBlockShape - blockInnerOffset, commBlockShape);
        } else {
            return;
        }
        
        auto tileShape = params.TileShape();
        EpilogueTileSwizzle epilogueTileSwizzle{actualCommBlockShape, tileShape};
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();

        for (uint32_t innerLoopIdx = 0; innerLoopIdx < tileLoops; innerLoopIdx++) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(innerLoopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;
            
            auto inTileOffset = inputBlockOffset + tileOffsetInBlock;
            auto outTileOffset = blockOutterOffset + blockInnerOffset + tileOffsetInBlock;

            // Get the data and layout of output
            auto gmSubblockC = gmC[layoutC.GetOffset(outTileOffset)];
            auto layoutSubblockC = layoutC.GetTileLayout(actualTileShape);

            // Get the data and layout of input
            AscendC::GlobalTensor<ElementDst> gmS;
            gmS.SetGlobalBuffer(reinterpret_cast<__gm__ ElementDst *>(params.shmemPtr));
            auto gmSubblockS = gmS[params.shmemLayout.GetOffset(inTileOffset)];
            auto layoutSubblockS = params.shmemLayout.GetTileLayout(actualTileShape);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
            tileRemoteCopy(
                gmSubblockC, layoutSubblockC,
                gmSubblockS, layoutSubblockS,
                actualTileShape,
                ubSList[ubListId],
                copyEventIdList[ubListId],
                rankIdx);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;
    AscendC::LocalTensor<ElementDst> ubSList[UB_STAGES];
    uint32_t copyEventIdList[UB_STAGES];
    uint32_t ubListId{0};
    TileRemoteCopy tileRemoteCopy;
};

} // namespace Catcoc::CommEpilogue::Block

#endif // CATCOC_COMM_EPILOGUE_BLOCK_EPILOGUE_TO_LOCAL_MEM_H