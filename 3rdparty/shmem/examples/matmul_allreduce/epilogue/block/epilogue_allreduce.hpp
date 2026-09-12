/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _EPILOGUE_ALLREDUCE_HPP
#define _EPILOGUE_ALLREDUCE_HPP

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

// from kernel
#include "epilogue/block/block_swizzle_dynamic.hpp"

// from shmem-device
#include "shmem.h"

using namespace Catlass;

CATLASS_DEVICE
MatrixCoord GetActualShape(const MatrixCoord &blockCount, const MatrixCoord &blockCoord, const MatrixCoord &blockShape,
                           const MatrixCoord &residue)
{
    MatrixCoord c = blockShape;

    if ((residue.row() != 0) && (blockCoord.row() == blockCount.row() - 1)) {
        c.row() = residue.row();
    } else if (blockCoord.row() >= blockCount.row()) {
        c.row() = 0;
    }

    if ((residue.column() != 0) && (blockCoord.column() == blockCount.column() - 1)) {
        c.column() = residue.column();
    } else if (blockCoord.column() >= blockCount.column()) {
        c.column() = 0;
    }
    return c;
}

namespace Catlass::Epilogue::Block {
template <class... Args>
class EpilogueAllReduce {};

template <class BlockScheduler_, class CommBlockSwizzle_>
class EpilogueAllReduce<BlockScheduler_, CommBlockSwizzle_> {
public:
    using BlockScheduler = BlockScheduler_;
    using CommBlockSwizzle = CommBlockSwizzle_;

    // Type aliases
    using ArchTag = Arch::AtlasA2;

    using ElementC = half;
    using ElementAttachedSource = half;
    using ElementAttachedOutput = half;

    using LayoutStore = layout::RowMajor;

    using LayoutWorkspace = LayoutStore;

    using ElementDestination = ElementAttachedOutput;

    using LayoutDestination = LayoutStore;

    using ScheduleTypeOp1 = typename Gemm::Block::ReduceScatterSchedule;
    using ScheduleTypeOp2 = typename Gemm::Block::AllGatherSchedule;

    // Epilogue params definition
    struct Params {
        AscendC::GlobalTensor<ElementDestination> destination;
        LayoutDestination layoutDestination;
        int64_t strideDestination;
        __gm__ ElementAttachedSource *symmetricPtr;
        BlockScheduler gemmSwizzle;
        CommBlockSwizzle commSwizzle;
        MatrixCoord blockShape;
        MatrixCoord processShape;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(AscendC::GlobalTensor<ElementDestination> destination, const LayoutDestination &layoutDestination,
               int64_t strideDestination, __gm__ ElementAttachedSource *symmetricPtr, MatrixCoord blockShape,
               MatrixCoord processShape, BlockScheduler gemmSwizzle, CommBlockSwizzle commSwizzle)
            : destination(destination),
              layoutDestination(layoutDestination),
              strideDestination(strideDestination),
              symmetricPtr(symmetricPtr),
              blockShape(blockShape),
              processShape(processShape),
              gemmSwizzle(gemmSwizzle),
              commSwizzle(commSwizzle)
        {
        }
    };

    CATLASS_DEVICE
    EpilogueAllReduce(Arch::Resource<ArchTag> &resource, Params const &params, const GemmCoord &blockGemmShape)
        : resource(resource), params(params), gemmBlockShape(blockGemmShape.m(), blockGemmShape.n())
    {
    }

    CATLASS_DEVICE
    ~EpilogueAllReduce()
    {
    }

    CATLASS_DEVICE
    void operator()(MatrixCoord const &blockShape, MatrixCoord const &commBlockCount,
                    MatrixCoord const &actualCommBlockCount, uint32_t calIdx, uint32_t rankIdx, uint32_t rankSize,
                    uint32_t pValue)
    {
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        int32_t aivIndex = AscendC::GetSubBlockIdx();
        auto loopNumPerComm = aicoreNum * pValue;

        auto layoutPeerMemStore =
            layout::RowMajor(blockShape.row() * loopNumPerComm * BufferNum, blockShape.column(), blockShape.column());

        // re-sliced Block by comm cores
        uint32_t flagIdx = calIdx % BufferNum;
        MatrixCoord actualCommBlockShape = blockShape * actualCommBlockCount;
        MatrixCoord outputBlockOffset = blockShape * MatrixCoord{calIdx * loopNumPerComm, 0};

        MatrixCoord commCoordPeerMem{flagIdx, 0};
        MatrixCoord blockOffset = commCoordPeerMem * commBlockCount * blockShape;

        params.commSwizzle.template SetProblemSize<ScheduleTypeOp1, true>(actualCommBlockShape);
        auto realAicoreNum = params.commSwizzle.GetRealCore();
        auto commCoreLoops = params.commSwizzle.GetCoreLoop();

        AscendC::GlobalTensor<ElementC> peerMem;
        peerMem.SetGlobalBuffer(params.symmetricPtr);

        // Local matmul is completed, waiting until tasks on all devices are complete.
        aclshmemx_barrier_all_vec();

        AscendC::SetAtomicAdd<ElementC>();
        AscendC::PipeBarrier<PIPE_ALL>();

        if (aivIndex == 0 && aicoreIndex < realAicoreNum) {  // only use one AIV core
            for (uint32_t idx = aicoreIndex; idx < commCoreLoops; idx += realAicoreNum) {
                MatrixCoord idxTile = params.commSwizzle.GetBlockIdx(idx);
                MatrixCoord actualCommSubBlockShape =
                    params.commSwizzle.template GetBlockSize<ScheduleTypeOp1>(idxTile);
                MatrixCoord rankBlockOffset = params.commSwizzle.template GetRankOffset<ScheduleTypeOp1>(idxTile);
                MatrixCoord subBlockOffset = params.commSwizzle.GetBlockOffset(idxTile);
                uint32_t mRankIdx = idxTile.column();
                if (mRankIdx == rankIdx) {
                    continue;
                }

                auto offsetIn = blockOffset + rankBlockOffset;
                auto offsetOut = blockOffset + rankBlockOffset;

                auto residueProcessShape = actualCommSubBlockShape % params.processShape;
                MatrixCoord processCount = CeilDiv(actualCommSubBlockShape, params.processShape);
                uint32_t processLoop = processCount.row() * processCount.column();

                // [ReduceScatter] 1. Alloc TmpUB
                int tmpBufferSize = 32 * 1024 / sizeof(ElementC);  // 32 KB
                AscendC::LocalTensor<half> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<ElementC>(0);
                tmpBuffer1.SetSize(tmpBufferSize);
                int tmpBufferOffset = 96 * 1024;  // half of UB
                AscendC::LocalTensor<half> tmpBuffer2 =
                    resource.ubBuf.template GetBufferByByte<ElementC>(tmpBufferOffset);
                tmpBuffer2.SetSize(tmpBufferSize);

                // [ReduceScatter] 2. Pre Interface Sync
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                int pingpongId = 0;

                for (uint32_t processIndex = 0; processIndex < processLoop; ++processIndex) {
                    AscendC::TEventID event_id = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<ElementC> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;

                    MatrixCoord processCoord{processIndex / processCount.column(),
                                             processIndex % processCount.column()};
                    auto actualProcessShape =
                        GetActualShape(processCount, processCoord, params.processShape, residueProcessShape);

                    auto processOffset = processCoord * params.processShape;

                    auto inputOffset = offsetIn + subBlockOffset + processOffset;
                    auto outputOffset = offsetOut + subBlockOffset + processOffset;

                    uint32_t copySize = actualProcessShape.row() * actualProcessShape.column();

                    int64_t inputElemOffset = layoutPeerMemStore.GetOffset(inputOffset);
                    int64_t outputElemOffset = layoutPeerMemStore.GetOffset(outputOffset);

                    // [ReduceScatter] 2. Pre Interface Sync
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

                    // [ReduceScatter] 3. Start aclshmemx_mte_get_nbi
                    aclshmemx_mte_get_nbi(peerMem[outputElemOffset], peerMem[inputElemOffset], buf, copySize,
                                          mRankIdx % rankSize, event_id);

                    // [ReduceScatter] 4. Post Interface Sync
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                    pingpongId = (pingpongId + 1) % BufferNum;
                }
                // [ReduceScatter] 4. Post Interface Sync
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);  // To SetAtomic, Scalar wait MTE3
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();

        // ReduceScatter is completed, waiting until tasks on all devices are complete.
        aclshmemx_barrier_all_vec();

        if (aivIndex == 0 && aicoreIndex < realAicoreNum) {
            for (uint32_t idx = aicoreIndex; idx < commCoreLoops; idx += realAicoreNum) {
                MatrixCoord idxTile = params.commSwizzle.GetBlockIdx(idx);
                MatrixCoord actualCommSubBlockShape =
                    params.commSwizzle.template GetBlockSize<ScheduleTypeOp2>(idxTile);
                MatrixCoord rankBlockOffset = params.commSwizzle.template GetRankOffset<ScheduleTypeOp2>(idxTile);
                MatrixCoord subBlockOffset = params.commSwizzle.GetBlockOffset(idxTile);

                uint32_t mRankIdx = idxTile.column();

                auto offsetIn = blockOffset + rankBlockOffset;
                auto offsetOut = outputBlockOffset + rankBlockOffset;

                auto residueProcessShape = actualCommSubBlockShape % params.processShape;
                MatrixCoord processCount = CeilDiv(actualCommSubBlockShape, params.processShape);

                uint32_t processLoop = processCount.row() * processCount.column();

                // [AllGather] 1. Alloc TmpUB
                int tmpBufferSize = 32 * 1024 / sizeof(ElementC);  // 32 KB
                AscendC::LocalTensor<half> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<ElementC>(0);
                tmpBuffer1.SetSize(tmpBufferSize);
                int tmpBufferOffset = 96 * 1024;  // half of UB
                AscendC::LocalTensor<half> tmpBuffer2 =
                    resource.ubBuf.template GetBufferByByte<ElementC>(tmpBufferOffset);
                tmpBuffer2.SetSize(tmpBufferSize);

                // [AllGather] 2. Pre Interface Sync
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                int pingpongId = 0;

                for (uint32_t processIndex = 0; processIndex < processLoop; ++processIndex) {
                    MatrixCoord processCoord{processIndex / processCount.column(),
                                             processIndex % processCount.column()};
                    auto actualProcessShape =
                        GetActualShape(processCount, processCoord, params.processShape, residueProcessShape);

                    auto processOffset = processCoord * params.processShape;

                    uint32_t residueM = actualProcessShape.row();

                    while (residueM > 0) {
                        MatrixCoord outputLoopOffset = (offsetOut + processOffset + subBlockOffset) / gemmBlockShape;
                        MatrixCoord residueOutputOffset = (offsetOut + processOffset + subBlockOffset) % gemmBlockShape;

                        auto loopIdx = outputLoopOffset.row();
                        GemmCoord outputBlockGemmTileOffset = params.gemmSwizzle.GetBlockCoord(loopIdx);
                        GemmCoord outputBlockGemmActualSize =
                            params.gemmSwizzle.GetActualBlockShape(outputBlockGemmTileOffset);
                        MatrixCoord outputBlockTileOffset{outputBlockGemmTileOffset.m(), outputBlockGemmTileOffset.n()};

                        uint32_t actualMoveM = min(gemmBlockShape.row() - residueOutputOffset.row(), residueM);
                        if (residueOutputOffset.row() < outputBlockGemmActualSize.m()) {
                            actualMoveM = min(outputBlockGemmActualSize.m() - residueOutputOffset.row(), residueM);

                            auto inputOffset = offsetIn + subBlockOffset + processOffset;
                            auto outputOffset = outputBlockTileOffset * gemmBlockShape + residueOutputOffset;

                            auto actualMoveShape = MatrixCoord{actualMoveM, outputBlockGemmActualSize.n()};
                            layout::RowMajor layoutInput = layoutPeerMemStore.GetTileLayout(actualMoveShape);
                            layout::RowMajor layoutOutput = params.layoutDestination.GetTileLayout(actualMoveShape);
                            int64_t inputElemOffset = layoutInput.GetOffset(inputOffset);
                            int64_t outputElemOffset = layoutOutput.GetOffset(outputOffset);

                            uint32_t copySize = actualMoveShape.row() * actualMoveShape.column();

                            AscendC::TEventID event_id = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
                            AscendC::LocalTensor<ElementC> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;

                            // [AllGather] 2. Pre Interface Sync
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

                            non_contiguous_copy_param copyParams;
                            copyParams.repeat = actualMoveShape.row();
                            copyParams.length = actualMoveShape.column();
                            copyParams.src_ld = layoutInput.stride(0);
                            copyParams.dst_ld = layoutOutput.stride(0);

                            // [AllGather] 3. Start aclshmemx_mte_get_nbi non-contiguous version
                            aclshmemx_mte_get_nbi(params.destination[outputElemOffset], peerMem[inputElemOffset], buf,
                                                  copyParams, mRankIdx % rankSize, event_id);

                            // [AllGather] 4. Post Interface Sync
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                            pingpongId = (pingpongId + 1) % BufferNum;
                        }
                        residueM -= actualMoveM;
                        processOffset += MatrixCoord{actualMoveM, 0};
                    }
                }
                // [AllGather] 4. Post Interface Sync
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }

        // AllGather is completed, waiting until tasks on all devices are complete.
        aclshmemx_barrier_all_vec();
    }

private:
    const static uint32_t BufferNum = 2;

    MatrixCoord gemmBlockShape;
    Arch::Resource<ArchTag> &resource;
    Params params;
};

}  // namespace Catlass::Epilogue::Block

#endif  // _EPILOGUE_ALLREDUCE_HPP