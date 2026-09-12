/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCOC_DGEMM_KERNEL_ALLGATHER_MATMUL_WITH_GATHER_RESULT_H
#define CATCOC_DGEMM_KERNEL_ALLGATHER_MATMUL_WITH_GATHER_RESULT_H

#include "catcoc/catcoc.h"

// from catlass
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catcoc::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    class BlockMmad_,
    class BlockEpilogueAllGather_,
    class BlockEpilogueCopyGatherA_,
    class BlockScheduler_,
    class BlockAllGatherScheduler_,
    class BlockCopyGatherAScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class AllGatherMatmulWithGatherResult {
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
    using ElementGatherA = ElementA;
    using LayoutGatherA = LayoutA;

    using AllGather = BlockEpilogueAllGather_;
    using AllGatherParams = typename AllGather::Params;

    using CopyGatherA = BlockEpilogueCopyGatherA_;
    using CopyGatherAParams = typename CopyGatherA::Params;

    using BlockScheduler = BlockScheduler_;
    using BlockAllGatherScheduler = BlockAllGatherScheduler_;
    using BlockCopyGatherAScheduler = BlockCopyGatherAScheduler_;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;

        uint32_t rankIdx;
        uint32_t rankSize;

        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        GM_ADDR ptrSymmetric;
        AllGatherParams allGatherParams;
        CopyGatherAParams copyGatherAParams;

        __gm__ ElementGatherA *ptrGatherA;
        LayoutGatherA layoutGatherA;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;

        uint32_t commInterval;

        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(
            GemmCoord const &problemShape_,
            uint32_t rank_, uint32_t rankSize_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrSymmetric_,
            AllGatherParams const &allGatherParams_,
            CopyGatherAParams const &copyGatherAParams_,
            GM_ADDR ptrGatherA_, LayoutGatherA const &layoutGatherA_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            uint32_t commInterval_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrSymmetric(ptrSymmetric_),
            allGatherParams(allGatherParams_),
            copyGatherAParams(copyGatherAParams_),
            ptrGatherA(reinterpret_cast<__gm__ ElementGatherA *>(ptrGatherA_)), layoutGatherA(layoutGatherA_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_),
            commInterval(commInterval_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    AllGatherMatmulWithGatherResult()
    {
        for (uint32_t stageIdx = 0; stageIdx< WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        GemmCoord blockShape = L1TileShape::ToCoord();
        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockMmad mmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementGatherA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementGatherA *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);

        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA))
        );
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        auto layoutC = params.layoutC;
        auto layoutCRowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        auto layoutCRow = layout::AffineRankN<3>(layoutCRowLogicStride);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualProblemShape = Catlass::MakeCoord<uint32_t>(
                actualCommSizeM, params.problemShape.n(), params.problemShape.k(), params.rankSize
            );
            BlockScheduler mmadScheduler(actualProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = mmadScheduler.GetCoreLoops();

            // wait aiv
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);

            for (uint32_t loopIdx = aicoreIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                auto blockOffset = mmadScheduler.GetBlockOffset(loopIdx);
                auto actualBlockShape = mmadScheduler.GetActualBlockShapeByOffset(blockOffset);

                uint32_t srcRankIdx = blockOffset.rank();
                MatrixCoord commOffsetA{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, srcRankIdx, 0)), 0};
                MatrixCoord commOffsetC{layoutCRow(Catlass::MakeCoord<int>(srcRankIdx, commIdx, 0)), 0};

                MatrixCoord offsetA = commOffsetA + blockOffset.GetCoordMK();
                MatrixCoord offsetB = blockOffset.GetCoordKN();
                MatrixCoord offsetC = commOffsetC + blockOffset.GetCoordMN();

                auto gmBlockA = gmSymmetric[layoutSymmetric.GetOffset(offsetA)];
                auto gmBlockB = gmB[params.layoutB.GetOffset(offsetB)];
                auto gmBlockC = gmC[layoutC.GetOffset(offsetC)];

                // Compute block-scoped matrix multiply-add
                mmad(
                    gmBlockA, layoutSymmetric,
                    gmBlockB, params.layoutB,
                    gmBlockC, layoutC,
                    actualBlockShape.GetCoordMNK()
                );
            }

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params &params)
    {
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        MatrixCoord blockShapeMK = MatrixCoord{L1TileShape::M, params.problemShape.k()};
        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        AllGather allGather(resource, params.allGatherParams);
        CopyGatherA copyGatherA(resource, params.copyGatherAParams);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementGatherA> gmGatherA;
        gmGatherA.SetGlobalBuffer(params.ptrGatherA);
        AscendC::GlobalTensor<ElementGatherA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementGatherA *>(params.ptrSymmetric));

        auto layoutGatherARowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        layout::AffineRankN<3> layoutGatherARow{layoutGatherARowLogicStride};
        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA))
        );
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        MatrixCoord commBlockShape = params.allGatherParams.BlockShape();
        MatrixCoord commCoreSplit = params.allGatherParams.CoreSplit();
        BlockAllGatherScheduler commScheduler(commBlockShape, commCoreSplit);

        MatrixCoord copyGatherABlockShape = params.copyGatherAParams.BlockShape();

        uint32_t copyCommIdx;
        uint32_t copyStageId;
        DistMatrixCoord copyActualShape;
        for (uint32_t commIdx = 0; commIdx < commLoops + 1; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, params.problemShape.k(), params.rankSize);
            MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
            commScheduler.UpdateProblem(actualCommShape, loopsInRank);
            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord commSrcOffset{commIdx * commSizeM, 0};
            MatrixCoord commDstOffset{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, params.rankIdx, 0)), 0};

            // wait aic
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            }

            aclshmemx_barrier_all_vec();

            if (commIdx < commLoops) {
                if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
                    allGather.InitBlockLoop();
                    for (uint32_t commLoopIdx = aicoreIdx; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                        DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                        MatrixCoord blockOffsetInRank = commScheduler
                                                        .GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                        MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);

                        uint32_t remoteRankIdx = commBlockCoord.rank();

                        auto offsetSrc = commSrcOffset + blockOffsetInRank;
                        auto offsetDst = commDstOffset + blockOffsetInRank;

                        auto gmBlockSrc = gmA[params.layoutA.GetOffset(offsetSrc)];
                        auto layoutBlockSrc = params.layoutA.GetTileLayout(actualCommBlockShape);

                        auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                        auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                        allGather(
                            gmBlockSrc, layoutBlockSrc,
                            gmBlockDst, layoutBlockDst,
                            actualCommBlockShape, remoteRankIdx % params.rankSize
                        );
                    }
                    allGather.FinalizeBlockLoop();
                }
            }

            if (commIdx > 0) {
                if (subcoreIdx == 1 && aicoreIdx >= commAicoreNum) {
                    BlockCopyGatherAScheduler copyGatherAScheduler{copyActualShape, copyGatherABlockShape};
                    uint32_t copyCoreLoops = copyGatherAScheduler.GetCoreLoops();
                    uint32_t copyAicoreNum = aicoreNum - commAicoreNum;
                    uint32_t copyAicoreIdx = aicoreIdx - commAicoreNum;

                    copyGatherA.InitBlockLoop();
                    for (uint32_t loopIdx = copyAicoreIdx; loopIdx < copyCoreLoops; loopIdx += copyAicoreNum) {
                        auto blockOffset = copyGatherAScheduler.GetBlockOffset(loopIdx);
                        auto actualBlockShape = copyGatherAScheduler.GetActualBlockShapeByOffset(blockOffset)
                                                .GetCoordInRank();

                        auto rowOffsetSrc = Catlass::MakeCoord<int>(copyStageId, blockOffset.rank(), blockOffset.row());
                        auto rowOffsetDst = Catlass::MakeCoord<int>(blockOffset.rank(), copyCommIdx, blockOffset.row());
                        MatrixCoord offsetSrc{
                            static_cast<uint32_t>(layoutSymmetricRow(rowOffsetSrc)), blockOffset.column()
                        };
                        MatrixCoord offsetDst{
                            static_cast<uint32_t>(layoutGatherARow(rowOffsetDst)), blockOffset.column()
                        };

                        auto gmBlockSrc = gmSymmetric[layoutSymmetric.GetOffset(offsetSrc)];
                        auto layoutBlockSrc = layoutSymmetric.GetTileLayout(actualBlockShape);

                        auto gmBlockDst = gmGatherA[params.layoutGatherA.GetOffset(offsetDst)];
                        auto layoutBlockDst = params.layoutGatherA.GetTileLayout(actualBlockShape);

                        copyGatherA(
                            gmBlockSrc, layoutBlockSrc,
                            gmBlockDst, layoutBlockDst,
                            actualBlockShape
                        );
                    }
                    copyGatherA.FinalizeBlockLoop();
                }
            }

            // BlockEpilogueAllGather is completed, waiting until tasks on all devices are complete.
            aclshmemx_barrier_all_vec();

            // set aic
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);

            copyCommIdx = commIdx;
            copyStageId = stageId;
            copyActualShape = actualCommShape;
        }
    }

private:
    // ID used for inter-core synchronization
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catcoc::Gemm::Kernel

#endif // CATCOC_DGEMM_KERNEL_ALLGATHER_MATMUL_WITH_GATHER_RESULT_H
