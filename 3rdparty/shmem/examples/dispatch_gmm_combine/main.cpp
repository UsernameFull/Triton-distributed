/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>

#include <iostream>
#include <vector>
#include <algorithm>

// misc
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "golden.hpp"
#include "opdev/fp16_t.h"

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "dispatch_policy_custom.h"

// from shmem-templates
#include "dispatch_gmm_combine.h"

// aclshmem_host
#include "host/shmem_host_def.h"
#include "host/mem/shmem_host_heap.h"
#include "host/init/shmem_host_init.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/team/shmem_host_team.h"
#include "shmem.h"
// utils
#include "utils.h"
#include "select_helper.h"

static uint32_t gNpuNum = 16;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catlass;
using fp16_t = op::fp16_t;

struct CoCTiling {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t swizzleDirect = 0;
    uint32_t swizzleOffset = 0;
    int32_t ubMoveNum = 0;
    uint32_t pValue = 0;
    uint32_t commNpuSplit = 0;
    uint32_t commDataSplit = 0;
    uint32_t lenPerLoop = 0;
    uint32_t EP = 0;
    uint32_t expertPerPe = 0;
    uint32_t maxOutputSize = 0;

    int64_t topK;
    int64_t activeNum;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensCountOrCumsumFlag;
    bool expertTokensBeforeCapacityFlag;
    int64_t quantMode;
    uint64_t initRoutingQuantTilingKey;
};

constexpr uint32_t
BLOCK_NUM = 8;
constexpr int32_t
BLOCK_SIZE_16 = 16;

template<class AType_,
    class BType_,
    class CType_,
    bool TB_,
    bool Nz_>
class DispatchGMMClass {
public:
    CATLASS_DEVICE
    DispatchGMMClass()
    {}

    CATLASS_DEVICE
    void Run(uint64_t fftsAddr, GemmCoord problemShape, GM_ADDR a, GM_ADDR b1, GM_ADDR b2, GM_ADDR c, GM_ADDR scale1,
             GM_ADDR scale2, GM_ADDR symmetricPtr,
             GM_ADDR expertIdx, GM_ADDR moeInitRoutingQuantV2Scale, GM_ADDR moeInitRoutingQuantV2Offset,
             GM_ADDR expertTokensBeforeCapacity, GM_ADDR probs,
             GM_ADDR ptrWorkspace, CoCTiling cocTiling,
             optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData)
    {
        // Define ArchTag
        using ArchTag = Arch::AtlasA2;
        constexpr bool enableUnitFlag = false;
        constexpr bool enableShuffleK = true;
        // unzip cocTiling
        uint32_t m = cocTiling.m;
        uint32_t n = cocTiling.n;
        uint32_t k = cocTiling.k;

        uint32_t epilogueCoreNum = 20;
        uint32_t epilogueGranularity = 17;

        uint32_t n2 = k;
        uint32_t k2 = n / 2;

        int32_t ubMoveNum = cocTiling.ubMoveNum;
        uint32_t EP = cocTiling.EP;
        uint32_t expertPerPe = cocTiling.expertPerPe;
        uint32_t maxOutputSize = cocTiling.maxOutputSize;
        int64_t activeNum = cocTiling.activeNum;
        int64_t expertCapacity = cocTiling.expertCapacity;
        int64_t expertNum = cocTiling.expertNum;
        int64_t dropPadMode = cocTiling.dropPadMode;
        int64_t expertTokensCountOrCumsumFlag = cocTiling.expertTokensCountOrCumsumFlag;
        bool expertTokensBeforeCapacityFlag = cocTiling.expertTokensBeforeCapacityFlag;
        int64_t quantMode = cocTiling.quantMode;
        int64_t topK = cocTiling.topK;
        uint64_t initRoutingQuantTilingKey = cocTiling.initRoutingQuantTilingKey;

        // Prepare comm address
        uint32_t rank = aclshmem_my_pe();
        uint32_t rankSize = aclshmem_n_pes();

        using LayoutA = layout::RowMajor;
        using LayoutB = typename std::conditional<
            Nz_,
            layout::zN,
            typename std::conditional<TB_, layout::ColumnMajor, layout::RowMajor>::type
        >::type;

        LayoutB layoutB1 = LayoutBInitializer<LayoutB, BType_>::create(k, n);
        LayoutB layoutB2 = LayoutBInitializer<LayoutB, BType_>::create(k2, n2);
        using LayoutC = layout::RowMajor;
        constexpr int L1TILEM = 128;
        constexpr int L1TILEN = 256;
        constexpr int L1TILEK = 512;
        using L1TileShape = GemmShape<L1TILEM, L1TILEN, L1TILEK>;   // M, N, K

        constexpr
        uint32_t workspaceStages = 2;
        constexpr
        uint32_t preloadStages = 1;
        constexpr
        uint32_t l1Stages = 2;
        constexpr
        uint32_t l0AStages = 2;
        constexpr
        uint32_t l0BStages = 2;
        constexpr
        uint32_t l0CStages = 1;
        constexpr
        uint32_t l1StagesNormal = 1;
        constexpr
        uint32_t l0BStagesNormal = 1;
        using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncFixpipe<
            preloadStages,
            l1Stages, l0AStages, l0BStages, l0CStages,
            enableUnitFlag, enableShuffleK
        >;

        constexpr int L0TILEM = 128;
        constexpr int L0TILEN = 256;
        constexpr int L0TILEK = 128;
        using L0TileShape = GemmShape<L0TILEM, L0TILEN, L0TILEK>;
        using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
        using BType = Gemm::GemmType<int8_t, LayoutB>;
        using CType = Gemm::GemmType<float16_t, layout::RowMajor>;
        using D1Type = Gemm::GemmType<int8_t, layout::RowMajor>;
        using D2Type = typename std::conditional<
            std::is_same_v<CType_, bfloat16_t>,
            Gemm::GemmType<bfloat16_t, layout::RowMajor>,
            Gemm::GemmType<CType_, layout::RowMajor>> ::type;

        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        constexpr
        uint32_t ubStages = 2;

        using EpilogueDispatchPolicy1 = Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>;

        using ScaleType = Gemm::GemmType<uint64_t, layout::VectorLayout>;
        using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
        using ElementMulType = Gemm::GemmType<float, layout::RowMajor>;
        using TileElemWiseMuls = Epilogue::Tile::TileElemWiseMuls<ArchTag, ElementMulType, 0>;

        using TileCopy1 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D1Type>;
        using BlockEpilogue1 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType,
                D1Type, TileElemWiseMuls, TileCopy1>;

        using EpilogueDispatchPolicy2 = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
        using TileCopy2 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
        using BlockEpilogue2 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType,
                D2Type, TileCopy2>;

        constexpr uint32_t SWIZZLE_GROUP_SIZE = 9;
        constexpr uint32_t SWIZZLE_DIRECTION = 1;
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<SWIZZLE_GROUP_SIZE, SWIZZLE_DIRECTION>;
        using ElementGroupList = int64_t;
        using MatmulKernel = Gemm::Kernel::DispatchGmmCombineKernel<BlockMmad,
                BlockScheduler, ElementGroupList, BlockEpilogue1, BlockEpilogue2>;

        LayoutA layoutA1{m, k};
        LayoutA layoutA2{m, k2};
        layout::VectorLayout layoutScale1{n};
        layout::VectorLayout layoutScale2{n2};
        layout::RowMajor layoutD1{maxOutputSize, k2};
        layout::RowMajor layoutD2{static_cast<uint32_t>(m * topK), n2};
        // Prepare params
        typename MatmulKernel::Params params{
            problemShape, cocTiling.EP, cocTiling.expertPerPe, cocTiling.maxOutputSize,
            rank, rankSize,
            activeNum, expertCapacity, expertNum, dropPadMode, expertTokensCountOrCumsumFlag,
            expertTokensBeforeCapacityFlag, quantMode, topK, initRoutingQuantTilingKey,
            epilogueCoreNum, epilogueGranularity,
            a, layoutA1, layoutA2,
            b1, layoutB1,
            b2, layoutB2,
            scale1, layoutScale1,
            scale2, layoutScale2,
            c, layoutD1, layoutD2,
            expertIdx, moeInitRoutingQuantV2Scale, moeInitRoutingQuantV2Offset,
            expertTokensBeforeCapacity, probs,
            ptrWorkspace,
            symmetricPtr, ubMoveNum, moeInitRoutingQuantV2TilingData};

        MatmulKernel kernel(params);
        kernel(params);
    }
};

CATLASS_GLOBAL
void DispatchGMM(
    uint64_t fftsAddr, GemmCoord problemShape, GM_ADDR a, GM_ADDR b1, GM_ADDR b2, GM_ADDR c, GM_ADDR scale1,
    GM_ADDR scale2, GM_ADDR symmetricPtr,
    GM_ADDR expertIdx, GM_ADDR moeInitRoutingQuantV2Scale, GM_ADDR moeInitRoutingQuantV2Offset,
    GM_ADDR expertTokensBeforeCapacity, GM_ADDR probs,
    GM_ADDR ptrWorkspace, CoCTiling cocTiling, int64_t tilingKey,
    optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(reinterpret_cast<uint64_t>(fftsAddr));

    DispatchGMMClass<int8_t, int8_t, float16_t, false, true> op;
    op.Run(fftsAddr, problemShape, a, b1, b2, c, scale1, scale2, symmetricPtr, expertIdx, moeInitRoutingQuantV2Scale,
        moeInitRoutingQuantV2Offset, expertTokensBeforeCapacity, probs,
        ptrWorkspace, cocTiling, moeInitRoutingQuantV2TilingData);
}

void InitData(uint8_t **hostPtr, uint8_t **devicePtr, size_t aSize, std::string path = "")
{
    std::cout << path << std::endl;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**> (devicePtr), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(hostPtr), aSize));
    if (path.length() == 0) {
        return;
    }
    ReadFile(path, *hostPtr, aSize);
    ACL_CHECK(aclrtMemcpy(*devicePtr, aSize, *hostPtr, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
}

aclshmemx_uniqueid_t default_flag_uid;

int main(int argc, char **argv)
{
    int status = ACLSHMEM_SUCCESS;
    int n_ranks = atoi(argv[1]);
    int rank_id = atoi(argv[2]);
    std::string ipport = argv[3];

    // Acl && Shmem init
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = atoi(argv[4]) + rank_id % gNpuNum;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    // status = aclshmemx_set_conf_store_tls(false, nullptr, 0);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, ipport.c_str(), default_flag_uid, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint32_t m = atoi(argv[5]);
    uint32_t k = atoi(argv[6]);
    uint32_t n = atoi(argv[7]);
    uint32_t EP = n_ranks;
    uint32_t expertPerPe = atoi(argv[8]);
    uint32_t dataType = atoi(argv[9]);
    uint32_t weightNz = atoi(argv[10]);
    uint32_t transB = atoi(argv[11]);
    uint32_t ubMoveNum = 3584;
    uint32_t topK = 8;
    uint32_t maxOutputSize = m * topK * 2;

    uint32_t k2 = n / 2;
    uint32_t n2 = k;

    // m, n, k
    GemmCoord problemShape{m, n, k};
    size_t aSize = static_cast<size_t>(m) * k * sizeof(float16_t);
    size_t b1Size = static_cast<size_t>(k) * n * expertPerPe * sizeof(int8_t);
    size_t b2Size = static_cast<size_t>(k2) * n2 * expertPerPe * sizeof(int8_t);
    size_t cSize = static_cast<size_t>(m) * n2 * sizeof(float16_t);
    size_t dequantScale1Size = static_cast<size_t>(expertPerPe) * n * sizeof(int64_t);
    size_t dequantScale2Size = static_cast<size_t>(expertPerPe) * n2 * sizeof(int64_t);
    size_t perTokenScaleSize = static_cast<size_t>(maxOutputSize) * sizeof(float);
    size_t probsSize = m * topK * sizeof(float);
    size_t lenTokenPerExpert = EP * EP * expertPerPe * sizeof(int32_t);

    uint32_t aivNum = 2 * BLOCK_NUM;

    size_t workspaceSize = m * topK * sizeof(int32_t) +
                           EP * EP * expertPerPe * sizeof(int32_t) * 3 +
                           maxOutputSize * sizeof(float32_t) * 2 +
                           std::max(maxOutputSize * n * sizeof(float16_t), maxOutputSize * n2 * sizeof(float16_t)) +
                           std::max(maxOutputSize * k * sizeof(int8_t), maxOutputSize * k2 * sizeof(int8_t));

    uint8_t *aDevice;
    uint8_t *aHost;
    uint8_t *b1Device;
    uint8_t *b1Host;
    uint8_t *b2Device;
    uint8_t *b2Host;
    uint8_t *cDevice;
    uint8_t *cHost;
    uint8_t *scale1Device;
    uint8_t *scale1Host;
    uint8_t *scale2Device;
    uint8_t *scale2Host;
    uint8_t *ptrWorkspace;
    uint8_t *probsDevice;
    uint8_t *probsHost;

    std::string filePrefix;
    const char *env_var = std::getenv("INPUT_PATH");
    if (env_var) {
        filePrefix = env_var;
    } else {
        std::cout << "请设置input 文件路径: export INPUT_PATH =" << std::endl;
    }
    std::string fileSuffix =
            "_" + std::to_string(dataType) + "_1_" + std::to_string(m) + "_" + std::to_string(k) + "_" +
            std::to_string(n) + "_" + std::to_string(expertPerPe) + "_" + std::to_string(EP) + "_1.bin";

    InitData(&b1Host, &b1Device, b1Size, filePrefix + "matrix_b1_" + std::to_string(rank_id) + fileSuffix);
    InitData(&b2Host, &b2Device, b2Size, filePrefix + "matrix_b2_" + std::to_string(rank_id) + fileSuffix);
    InitData(&cHost, &cDevice, cSize);
    InitData(&scale1Host, &scale1Device, dequantScale1Size,
             filePrefix + "matrix_dequant_scale1_" + std::to_string(rank_id) + fileSuffix);
    InitData(&scale2Host, &scale2Device, dequantScale2Size,
             filePrefix + "matrix_dequant_scale2_" + std::to_string(rank_id) + fileSuffix);
    InitData(&probsHost, &probsDevice, probsSize, filePrefix + "probs" + fileSuffix);

    uint8_t *expertIdx;
    uint8_t *expertIdxHost;
    uint8_t *moeInitRoutingQuantV2Scale;
    uint8_t *moeInitRoutingQuantV2Offset;
    uint8_t *expandedX;
    uint8_t *expandedXHost;

    uint8_t *expertTokensBeforeCapacity;
    int64_t activeNum = 0;
    int64_t expertCapacity = 0;
    int64_t expertNum = expertPerPe * EP;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantMode = 1;
    std::string dispatchFileSuffix = "";
    InitData(&aHost, &aDevice, m * k * sizeof(float16_t),
             filePrefix + "matrix_a_" + std::to_string(rank_id) + fileSuffix);
    InitData(&expertIdxHost, &expertIdx, m * topK * sizeof(int32_t),
             filePrefix + "expert_idx_" + std::to_string(rank_id) + fileSuffix);

    moeInitRoutingQuantV2Scale = nullptr;
    moeInitRoutingQuantV2Offset = nullptr;
    expertTokensBeforeCapacity = nullptr;

    optiling::MoeInitRoutingQuantV2TilingBase moeInitRoutingQuantV2TilingBase;
    int64_t inuptXDtypeSize = sizeof(float16_t);
    int64_t scaleDim0 = 0;
    int64_t ubSize = 196352;
    moeInitRoutingQuantV2TilingBase.DoTiling(m, k, topK, expertCapacity, expertNum, activeNum, dropPadMode,
                                             expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
                                             inuptXDtypeSize, quantMode, scaleDim0, aivNum, ubSize);
    uint64_t initRoutingQuantTilingKey = moeInitRoutingQuantV2TilingBase.tilingKey_;
    size_t initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;
    workspaceSize += initRoutingWorkspace;
    printf("!!!!!!!!!! initRoutingQuantTilingKey %lu\n\n", initRoutingQuantTilingKey);
    if (rank_id == 0) {
        moeInitRoutingQuantV2TilingBase.ShowTilingData();
    }

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&ptrWorkspace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    int32_t aclshmem_size = (504 * 1024 * 1024) * sizeof(__fp16);
    void *symmPtr = aclshmem_malloc(aclshmem_size);
    uint8_t *symmetricPtr = (uint8_t *) symmPtr;

    CoCTiling cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    cocTiling.ubMoveNum = ubMoveNum;
    cocTiling.maxOutputSize = maxOutputSize;
    cocTiling.EP = EP;
    cocTiling.expertPerPe = expertPerPe;
    cocTiling.activeNum = activeNum;
    cocTiling.expertCapacity = expertCapacity;
    cocTiling.expertNum = expertNum;
    cocTiling.dropPadMode = dropPadMode;
    cocTiling.expertTokensCountOrCumsumFlag = expertTokensCountOrCumsumFlag;
    cocTiling.expertTokensBeforeCapacityFlag = expertTokensBeforeCapacityFlag;
    cocTiling.quantMode = quantMode;
    cocTiling.topK = topK;
    cocTiling.initRoutingQuantTilingKey = initRoutingQuantTilingKey;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    for (int i = 0; i < 1; ++i) {
        uint64_t fftsAddr = util_get_ffts_config();
        ACL_CHECK(aclrtMemcpy(b1Device, b1Size, b1Host, b1Size, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(b2Device, b2Size, b2Host, b2Size, ACL_MEMCPY_HOST_TO_DEVICE));
        DispatchGMM<<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, problemShape, aDevice, b1Device, b2Device,
                cDevice, scale1Device, scale2Device, symmetricPtr,
                expertIdx, moeInitRoutingQuantV2Scale, moeInitRoutingQuantV2Offset,
                expertTokensBeforeCapacity, probsDevice,
                ptrWorkspace, cocTiling, 0, moeInitRoutingQuantV2TilingBase.quantTilingData);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./out/output_" + std::to_string(rank_id) + ".bin", cHost, cSize);
    if (rank_id == 0) {
        std::printf("\ntest finished\n");
    }
    aclshmem_free(symmPtr);
    ACL_CHECK(aclrtFreeHost(b1Host));
    ACL_CHECK(aclrtFreeHost(b2Host));
    ACL_CHECK(aclrtFreeHost(cHost));
    ACL_CHECK(aclrtFree(b1Device));
    ACL_CHECK(aclrtFree(b2Device));
    ACL_CHECK(aclrtFree(cDevice));
    ACL_CHECK(aclrtFreeHost(expertIdxHost));
    ACL_CHECK(aclrtFree(expertIdx));

    status = aclrtDestroyStream(stream);

    status = aclshmem_finalize();
    status = aclrtResetDevice(deviceId);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in pe_id " << rank_id << std::endl;
    return 0;
}
