/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCOC_COMM_EPILOGUE_BLOCK_SWIZZLE_H
#define CATCOC_COMM_EPILOGUE_BLOCK_SWIZZLE_H

#include "catcoc/catcoc.h"
#include "catcoc/detail/remote_copy_type.h"
#include "catcoc/dist_coord.h"

// from catlass
#include "catlass/detail/alignment.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catcoc::CommEpilogue::Block {

using Catlass::MatrixCoord;

template<uint32_t SWIZZLE_DIRECTION_ = 0, bool IS_DETERMINISTIC_ = false>
struct BlockCommSwizzle {
    static constexpr uint32_t SWIZZLE_DIRECTION = SWIZZLE_DIRECTION_;
    static constexpr uint32_t IS_DETERMINISTIC = IS_DETERMINISTIC_;

    static_assert((IS_DETERMINISTIC && SWIZZLE_DIRECTION == 0) || !IS_DETERMINISTIC,
        "Deterministic calculation requires that the swizzle direction be 0.");

    DistMatrixCoord problemShape;
    DistMatrixCoord loops;

    uint32_t swizzleOffset;
    MatrixCoord coreSplit;
    DistMatrixCoord blockShape;

    CATLASS_DEVICE
    BlockCommSwizzle() {}

    CATLASS_DEVICE
    BlockCommSwizzle(DistMatrixCoord const &problemShape_, MatrixCoord const &blockShapeInRank_,
        MatrixCoord const &coreSplit_, MatrixCoord const &loopsInRank_)
        : problemShape(problemShape_), coreSplit(coreSplit_)
    {
        blockShape = Catlass::MakeCoord<uint32_t>(blockShapeInRank_.row(), blockShapeInRank_.column(), 1);
        loops = Catlass::MakeCoord<uint32_t>(loopsInRank_.row(), loopsInRank_.column(), problemShape_.rank());

        if constexpr (IS_DETERMINISTIC) {
            coreSplit = MatrixCoord{coreSplit.row() * coreSplit.column(), 1};
        }

        if constexpr (SWIZZLE_DIRECTION == 0) {
            swizzleOffset = coreSplit.row();
        } else {
            swizzleOffset = coreSplit.column();
        }
    }

    CATLASS_DEVICE
    BlockCommSwizzle(MatrixCoord const &blockShapeInRank_, MatrixCoord const &coreSplit_)
        : coreSplit(coreSplit_)
    {
        blockShape = Catlass::MakeCoord<uint32_t>(blockShapeInRank_.row(), blockShapeInRank_.column(), 1);

        if constexpr (IS_DETERMINISTIC) {
            coreSplit = MatrixCoord{coreSplit.row() * coreSplit.column(), 1};
        }

        if constexpr (SWIZZLE_DIRECTION == 0) {
            swizzleOffset = coreSplit.row();
        } else {
            swizzleOffset = coreSplit.column();
        }
    }

    CATLASS_DEVICE
    BlockCommSwizzle(DistMatrixCoord const &blockShape_, MatrixCoord const &coreSplit_)
        : blockShape(blockShape_), coreSplit(coreSplit_)
    {
        if constexpr (IS_DETERMINISTIC) {
            coreSplit = MatrixCoord{coreSplit.row() * coreSplit.column(), 1};
        }

        if constexpr (SWIZZLE_DIRECTION == 0) {
            swizzleOffset = coreSplit.row();
        } else {
            swizzleOffset = coreSplit.column();
        }
    }

    CATLASS_DEVICE
    void UpdateProblem(DistMatrixCoord const &problemShape_, DistMatrixCoord const &loops_)
    {
        problemShape = problemShape_;
        loops = loops_;
    }

    CATLASS_DEVICE
    void UpdateProblem(DistMatrixCoord const &problemShape_, MatrixCoord const &loopsInRank_)
    {
        problemShape = problemShape_;
        loops = Catlass::MakeCoord<uint32_t>(loopsInRank_.row(), loopsInRank_.column(), problemShape_.rank());
    }

    CATLASS_DEVICE
    void UpdateProblem(DistMatrixCoord const &problemShape_)
    {
        problemShape = problemShape_;
        loops = CeilDiv(problemShape, blockShape);
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoop() const
    {
        if constexpr (IS_DETERMINISTIC) {
            return RoundUp<uint32_t>(loops.row() * loops.column(), coreSplit.row()) * loops.rank();
        } else {
            return loops.row() * loops.column() * loops.rank();
        }
    }

    CATLASS_DEVICE
    uint32_t GetRealCore() const
    {
        return coreSplit.row() * coreSplit.column();
    }

    CATLASS_DEVICE
    DistMatrixCoord GetBlockCoord(uint32_t taskIdx) const
    {
        uint32_t innerIdx = taskIdx % GetCoreLoop();
        uint32_t dataLoops = loops.row() * loops.column();
        uint32_t rankLoops = loops.rank();
        uint32_t nStride = loops.rank() / coreSplit.column();
        if (rankLoops == 0) {
            DistMatrixCoord{};
        }

        if constexpr (SWIZZLE_DIRECTION == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(dataLoops, swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * rankLoops);
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * rankLoops);

            uint32_t nRow = swizzleOffset;
            if constexpr (!IS_DETERMINISTIC) {
                if (tileBlockIdx == tileBlockLoop - 1) {
                    nRow = dataLoops - swizzleOffset * tileBlockIdx;
                }
            }
            uint32_t dataIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nRow;
            uint32_t rankIdx = inTileBlockIdx / nRow;

            rankIdx = (rankIdx * nStride) % rankLoops + (rankIdx * nStride) / rankLoops;
            rankIdx = (rankIdx + dataIdx) % rankLoops;

            return DistMatrixCoord{dataIdx / loops.column(), dataIdx % loops.column(), rankIdx};
        } else if (SWIZZLE_DIRECTION == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(rankLoops, swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * dataLoops);
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * dataLoops);

            uint32_t nCol = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = rankLoops - swizzleOffset * tileBlockIdx;
            }
            uint32_t dataIdx = inTileBlockIdx / nCol;
            uint32_t rankIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nCol;

            rankIdx = (rankIdx * nStride) % rankLoops + (rankIdx * nStride) / rankLoops;
            rankIdx = (rankIdx + dataIdx) % rankLoops;

            return DistMatrixCoord{dataIdx / loops.column(), dataIdx % loops.column(), rankIdx};
        }
        return DistMatrixCoord{};
    }

    CATLASS_DEVICE
    MatrixCoord GetBlockOffset(DistMatrixCoord blockCoord) const
    {
        if (blockCoord.rank() >= loops.rank()
            || blockCoord.row() >= loops.row()
            || blockCoord.column() >= loops.column()) {
            return MatrixCoord{UINT_MAX, UINT_MAX};
        }

        auto layoutRowLogicShape = Catlass::MakeCoord<uint32_t>(loops.rank(), loops.row());
        auto layoutRowExpandRank = layout::AffineRankN<2, uint32_t>::Packed(layoutRowLogicShape);
        uint32_t rowCoordPostRank = layoutRowExpandRank(
            Catlass::MakeCoord<uint32_t>(blockCoord.rank(), blockCoord.row()));
        return MatrixCoord{rowCoordPostRank, blockCoord.column()} * blockShape.GetCoordInRank();
    }

    CATLASS_DEVICE
    MatrixCoord GetBlockOffsetInRank(MatrixCoord blockCoordInRank) const
    {
        if (blockCoordInRank.row() >= loops.row() || blockCoordInRank.column() >= loops.column()) {
            return MatrixCoord{UINT_MAX, UINT_MAX};
        }
        return blockCoordInRank * blockShape.GetCoordInRank();
    }

    CATLASS_DEVICE
    MatrixCoord GetActualBlockShapeByOffset(MatrixCoord blockOffset)
    {
        auto residue = problemShape.GetCoordInRank() - Min<uint32_t, 2>(problemShape.GetCoordInRank(), blockOffset);
        auto actualBlockShape = Min(blockShape.GetCoordInRank(), residue);
        return actualBlockShape;
    }

    CATLASS_DEVICE
    MatrixCoord GetActualBlockShape(MatrixCoord blockCoordInRank) const
    {
        auto blockOffset = GetBlockOffsetInRank(blockCoordInRank);
        auto residue = problemShape.GetCoordInRank() - Min<uint32_t, 2>(problemShape.GetCoordInRank(), blockOffset);
        auto actualBlockShape = Min(blockShape.GetCoordInRank(), residue);
        return actualBlockShape;
    }
};

struct BlockSchedulerCopyGatherA {
    DistMatrixCoord problemShape;
    DistMatrixCoord tileShape;
    DistMatrixCoord gridShape;

    CATLASS_DEVICE
    BlockSchedulerCopyGatherA() = default;

    CATLASS_DEVICE
    BlockSchedulerCopyGatherA(DistMatrixCoord const &problemShape_, DistMatrixCoord const &tileShape_)
        : problemShape(problemShape_), tileShape(tileShape_)
    {
        gridShape = CeilDiv(problemShape, tileShape);
    }

    CATLASS_DEVICE
    BlockSchedulerCopyGatherA(DistMatrixCoord const &problemShape_, MatrixCoord const &tileShapeMN_)
        : BlockSchedulerCopyGatherA(problemShape_, DistMatrixCoord{tileShapeMN_.row(), tileShapeMN_.column(), 1})
    {
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return Numel(gridShape);
    }

    CATLASS_DEVICE
    DistMatrixCoord GetBlockCoord(uint32_t loopIdx) const
    {
        uint32_t dataLoops = Numel(gridShape.GetCoordInRank());
        if (dataLoops == 0) {
            return DistMatrixCoord{};
        }
        uint32_t rankIdx = loopIdx / dataLoops;
        uint32_t dataIdx = loopIdx % dataLoops;
        return {dataIdx / gridShape.column(), dataIdx % gridShape.column(), rankIdx};
    }

    CATLASS_DEVICE
    DistMatrixCoord GetBlockOffset(uint32_t loopIdx) const
    {
        return GetBlockCoord(loopIdx) * tileShape;
    }

    CATLASS_DEVICE
    DistMatrixCoord GetActualBlockShapeByOffset(DistMatrixCoord const &blockOffset) const
    {
        return Min(tileShape, problemShape - blockOffset);
    }

    CATLASS_DEVICE
    DistMatrixCoord GetActualBlockShape(DistMatrixCoord const &blockCoord) const
    {
        auto blockOffset = blockCoord * tileShape;
        return GetActualBlockShapeByOffset(blockOffset);
    }
};

}  // namespace Catcoc::CommEpilogue::Block

#endif // CATCOC_COMM_EPILOGUE_BLOCK_SWIZZLE_H