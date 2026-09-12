/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCOC_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_H
#define CATCOC_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_H

#include "catcoc/catcoc.h"

#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catcoc::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    class BlockMmad_,
    class BlockEpilogueReduceScatter_,
    class BlockMmadScheduler_,
    class BlockEpilogueScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class MatmulReduceScatter {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockEpilogueReduceScatter = BlockEpilogueReduceScatter_;
    using BlockEpilogueReduceScatterParams = typename BlockEpilogueReduceScatter::Params;

    using ElementD = typename BlockEpilogueReduceScatter::ElementDst;
    using LayoutD = typename BlockEpilogueReduceScatter::LayoutDst;

    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockEpilogueScheduler = BlockEpilogueScheduler_;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    struct Params {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;

        uint32_t commInterval;

        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementD *ptrD;
        LayoutD layoutD;
        GM_ADDR ptrSymmetric;

        BlockEpilogueReduceScatterParams reduceScatterParams;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(
            GemmCoord const &problemShape_, uint32_t rankIdx_, uint32_t rankSize_,
            uint32_t commInterval_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrD_, LayoutD const &layoutD_,
            GM_ADDR ptrSymmetric_,
            BlockEpilogueReduceScatterParams const &reduceScatterParams_
        ) : problemShape(problemShape_), rankIdx(rankIdx_), rankSize(rankSize_),
            commInterval(commInterval_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)), layoutD(layoutD_),
            ptrSymmetric(ptrSymmetric_),
            reduceScatterParams(reduceScatterParams_)
        {
        }
    };

    CATLASS_DEVICE
    MatmulReduceScatter()
    {
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

        GemmCoord blockShape = L1TileShape::ToCoord();
        GemmCoord problemShapeInRank = params.problemShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1, 1);
        BlockMmadScheduler mmadScheduler(problemShapeInRank, blockShape.GetCoordMN());
        uint32_t coreLoops = mmadScheduler.GetCoreLoops() * params.rankSize;
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(params.ptrD));

        auto layoutSymmetric = Catlass::layout::RowMajor{
            WORKSPACE_STAGES * blockPerComm * L1TileShape::M, L1TileShape::N,
            L1TileShape::N
        };

        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, blockPerComm, L1TileShape::M);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;

            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageIdx]);
            }

            uint32_t actualBlockPerComm = (commIdx == commLoops - 1) ?
                (coreLoops - blockPerComm * commIdx) : blockPerComm;
            uint32_t actualBlockPerCommInRank = actualBlockPerComm / params.rankSize;

            uint32_t commBlockOffsetInRank = commIdx * blockPerCommInRank;
            for (uint32_t blockIdxInComm = aicoreIdx; blockIdxInComm < actualBlockPerComm;
                blockIdxInComm += aicoreNum) {
                uint32_t loopIdxInRank = commBlockOffsetInRank + blockIdxInComm % actualBlockPerCommInRank;
                uint32_t targetRankIdx = blockIdxInComm / actualBlockPerCommInRank;
                GemmCoord blockCoord = mmadScheduler.GetBlockCoord(loopIdxInRank);
                GemmCoord actualBlockShape = mmadScheduler.GetActualBlockShape(blockCoord);

                GemmCoord offsetCoord = blockCoord * blockShape;
                auto rankOffsetA = problemShapeInRank.GetCoordMK() * Catlass::MakeCoord<uint32_t>(targetRankIdx, 0);
                auto blockOffsetA = offsetCoord.GetCoordMK() + rankOffsetA;
                auto blockOffsetB = offsetCoord.GetCoordKN();

                auto gmBlockA = gmA[params.layoutA.GetOffset(blockOffsetA)];
                auto gmBlockB = gmB[params.layoutB.GetOffset(blockOffsetB)];

                AscendC::GlobalTensor<ElementC> gmBlockC;
                Catlass::layout::RowMajor layoutC;
                if (targetRankIdx == params.rankIdx) {
                    MatrixCoord blockOffsetD = offsetCoord.GetCoordMN();
                    gmBlockC = gmD[params.layoutD.GetOffset(blockOffsetD)];
                    layoutC = params.layoutD;
                }
                else {
                    MatrixCoord blockOffsetSymmetric = MatrixCoord{
                        layoutSymmetricRow(Catlass::MakeCoord<int>(stageIdx, blockIdxInComm, 0)), 0
                    };
                    gmBlockC = gmSymmetric[layoutSymmetric.GetOffset(blockOffsetSymmetric)];
                    layoutC = layoutSymmetric;
                }

                blockMmad(
                    gmBlockA, params.layoutA,
                    gmBlockB, params.layoutB,
                    gmBlockC, layoutC,
                    actualBlockShape
                );
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

        MatrixCoord blockShapeMN = L1TileShape::ToCoordMN();
        GemmCoord problemShapeInRank = params.problemShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1, 1);
        BlockMmadScheduler mmadScheduler(problemShapeInRank, blockShapeMN);
        uint32_t coreLoops = mmadScheduler.GetCoreLoops() * params.rankSize;
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

        BlockEpilogueReduceScatter reduceScatter(resource, params.reduceScatterParams);

        AscendC::GlobalTensor<ElementC> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
        auto layoutSymmetric = Catlass::layout::RowMajor{
            WORKSPACE_STAGES * blockPerComm * L1TileShape::M, L1TileShape::N,
            L1TileShape::N
        };

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        MatrixCoord commBlockShape = params.reduceScatterParams.BlockShape();
        MatrixCoord commCoreSplit = params.reduceScatterParams.CoreSplit();
        BlockEpilogueScheduler commScheduler(commBlockShape, commCoreSplit);
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            uint32_t actualBlockInComm = Min(blockPerComm, coreLoops - commIdx * blockPerComm);
            auto actualCommShape = DistMatrixCoord{actualBlockInComm * blockShapeMN.row() / params.rankSize,
                blockShapeMN.column(), params.rankSize};
            MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);

            commScheduler.UpdateProblem(actualCommShape, loopsInRank);
            uint32_t commAicoreNum = commScheduler.GetRealCore();
            uint32_t commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord stageOffset = MatrixCoord{stageIdx * blockPerComm, 0} * blockShapeMN;
            uint32_t mmadStartLoopIdxInComm = commIdx * blockPerCommInRank;

            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);

            aclshmemx_barrier_all_vec();

            AscendC::SetAtomicAdd<ElementD>();
            AscendC::PipeBarrier<PIPE_ALL>();
            reduceScatter.InitBlockLoop();
            if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIdx; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                    MatrixCoord blockOffset = commScheduler.GetBlockOffset(
                        DistMatrixCoord{commBlockCoord.GetCoordInRank(), params.rankIdx});
                    MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                    MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);

                    uint32_t remoteRankIdx = commBlockCoord.rank();
                    if (remoteRankIdx == params.rankIdx) {
                        continue;
                    }

                    uint32_t mmadLoopIdx = mmadStartLoopIdxInComm + blockOffsetInRank.row() / blockShapeMN.row();
                    GemmCoord mmadBlockCoordMNK = mmadScheduler.GetBlockCoord(mmadLoopIdx);
                    MatrixCoord mmadBlockCoord = mmadBlockCoordMNK.GetCoordMN();
                    MatrixCoord actualMmadBlockShape =
                        mmadScheduler.GetActualBlockShape(mmadBlockCoordMNK).GetCoordMN();

                    MatrixCoord offsetInMmadBlock = blockOffsetInRank % blockShapeMN;
                    MatrixCoord residueInMmadBlock = actualMmadBlockShape -
                        Min<uint32_t, 2>(actualMmadBlockShape, offsetInMmadBlock);
                    actualCommBlockShape = Min<uint32_t, 2>(actualCommBlockShape, residueInMmadBlock);

                    auto offsetSrc = stageOffset + blockOffset;
                    MatrixCoord mmadBlockOffset = mmadBlockCoord * blockShapeMN;
                    auto offsetDst = mmadBlockOffset + offsetInMmadBlock;
                    
                    auto gmBlockSrc = gmSymmetric[layoutSymmetric.GetOffset(offsetSrc)];
                    auto layoutBlockSrc = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                    auto gmBlockDst = gmD[params.layoutD.GetOffset(offsetDst)];
                    auto layoutBlockDst = params.layoutD.GetTileLayout(actualCommBlockShape);

                    reduceScatter(
                        gmBlockSrc, layoutBlockSrc,
                        gmBlockDst, layoutBlockDst,
                        actualCommBlockShape, remoteRankIdx % params.rankSize
                    );
                }
            }
            reduceScatter.FinalizeBlockLoop();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone();
            AscendC::PipeBarrier<PIPE_ALL>();

            aclshmemx_barrier_all_vec();

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageIdx]);
        }
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

}  // namespace Catcoc::DGemm::Kernel

#endif  // CATCOC_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_H
