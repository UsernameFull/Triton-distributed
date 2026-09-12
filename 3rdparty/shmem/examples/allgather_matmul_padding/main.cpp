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
#include <cstring>
#include <algorithm>

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

// aclshmem_host
#include "host/shmem_host_def.h"
#include "host/mem/shmem_host_heap.h"
#include "host/init/shmem_host_init.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/team/shmem_host_team.h"

// utils
#include "utils.h"

#include "catcoc/catcoc.h"
#include "catcoc/comm_epilogue/comm_dispatch_policy.h"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.h"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.h"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.h"
#include "catcoc/detail/remote_copy_type.h"
#include "catcoc/dgemm/block/block_swizzle_allgather.h"
#include "catcoc/dgemm/kernel/allgather_matmul_padding.h"

static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catcoc;

constexpr uint32_t BLOCK_NUM = 20;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;

using ElementA = half;
using ElementB = half;
using ElementC = half;

template <bool PADDING_B>
CATLASS_GLOBAL
void ShmemAllGatherMatmulPadding(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC,
    GM_ADDR workSpace, GM_ADDR gmSymmetric,
    uint32_t m, uint32_t n, uint32_t k)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    // Define ArchTag
    using ArchTag = Catlass::Arch::AtlasA2;

    // Prepare comm address
    uint32_t pe = aclshmem_my_pe();
    uint32_t peSize = aclshmem_n_pes();

    Catlass::GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m * peSize, n};

    // Block level, define BlockMmad
    constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using BlockMmadScheduler = typename Catcoc::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0>;

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Put>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommBlockShape = Catlass::MatrixShape<64, UINT_MAX / 2>;
    using CommCoreSplit = Catlass::MatrixShape<20, 1>;

    constexpr uint32_t alignByByte = 512;
    constexpr uint32_t alignByElement = alignByByte / sizeof(ElementA);

    bool isNeedPaddingB = Catcoc::Padding::IsNeedPadding(layoutB, alignByElement);
    using PaddingHelperB = typename Catcoc::Padding::PaddingHelper<BType, PADDING_B>;
    using LayoutWB = typename PaddingHelperB::LayoutW;
    LayoutWB layoutWB = PaddingHelperB::GetLayoutW(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
    using ActualTypeB = typename PaddingHelperB::ActualType;
    using GlobalPaddingB = typename PaddingHelperB::GlobalPadding;

    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, ActualTypeB, CType
    >;

    constexpr uint32_t UB_STAGES = 2;
    using EpilogueAllGatherTileShape = Catlass::MatrixShape<32, 256>;
    using EpilogueAllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
        Catcoc::detail::CopyMode::Gather>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<
        EpilogueAllGatherDispatch,
        RemoteSrcType, RemoteDstType,
        CommCoreSplit,
        CommBlockShape,
        EpilogueAllGatherTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using AllGatherMatmulKernel = DGemm::Kernel::AllGatherMatmulPadding<
        GlobalPaddingB,
        BlockMmad,
        BlockEpilogueAllGather,
        BlockMmadScheduler,
        BlockEpilogueScheduler,
        WORKSPACE_STAGES
    >;

    typename BlockEpilogueAllGather::Params allGatherParams{};

    // Prepare params
    typename AllGatherMatmulKernel::Params params{
        problemShape,
        pe, peSize,
        COMM_INTERVAL,
        gmA, layoutA,
        gmB, layoutB,
        gmC, layoutC,
        workSpace, layoutWB,
        gmSymmetric,
        allGatherParams
    };

    // Call kernel
    AllGatherMatmulKernel matmulCommKernel;
    matmulCommKernel(params);
}

struct Options {
    static constexpr auto HELPER =
       "Usage: allgather_matmul_padding pe_size pe_id ip_port m n k [device_id_list]\n";

    int peSize;
    int peId;
    std::string ipPort;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::string dataPath;
    std::vector<int> deviceIdList{};

    int Parse(int argc, char **argv)
    {
        enum ArgsIndex {
            PE_SIZE_INDEX = 1,
            PE_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DATA_PATH_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };

        if (argc > INDEX_MAX) {
            printf(HELPER);
            return -1;
        }

        peSize = std::atoi(argv[PE_SIZE_INDEX]);
        peId = std::atoi(argv[PE_ID_INDEX]);
        ipPort = argv[IP_PORT_INDEX];
        m = std::atoi(argv[M_INDEX]);
        n = std::atoi(argv[N_INDEX]);
        k = std::atoi(argv[K_INDEX]);
        dataPath = argv[DATA_PATH_INDEX];
        if (argc > DEVICE_LIST_INDEX) {
            char *idListStr = argv[DEVICE_LIST_INDEX];
            for (char *idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (size_t i = 0; i < peSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        return 0;
    }

    std::string GetDataPath(std::string const &fileName = "") const
    {
        return dataPath + "/" + fileName;
    }
};

aclshmemx_uniqueid_t default_flag_uid;

int main(int argc, char **argv)
{
    int status = ACLSHMEM_SUCCESS;
    // Kernel-need params parse
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }

    int n_pes = options.peSize;
    int pe_id = options.peId;
    std::string ipPort = options.ipPort;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    int32_t device_id = options.deviceIdList[pe_id];

    // Acl && Shmem init
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(pe_id, n_pes, local_mem_size, ipPort.c_str(), default_flag_uid, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // ACLStream init
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);

    std::cout << "[TEST] input pe_size: " << n_pes << " pe_id:" << pe_id << std::endl;

    LayoutB layoutB{k, n};
    constexpr uint32_t alignByByte = 512;
    constexpr uint32_t alignByElement = alignByByte / sizeof(ElementA);

    bool isNeedPaddingB = Catcoc::Padding::IsNeedPadding(layoutB, alignByElement);

    constexpr int TILE_N = 128;
    constexpr int TILE_M = 256;
    constexpr int TILE_K = 256;
    using L1TileShape =
        std::conditional_t<std::is_same_v<LayoutA, Catlass::layout::ColumnMajor> && std::is_same_v<LayoutB,
                           Catlass::layout::ColumnMajor>,
                           Catlass::GemmShape<TILE_M, TILE_N, TILE_K>, Catlass::GemmShape<TILE_N, TILE_M, TILE_K>>;

    size_t aSize = static_cast<size_t>(m) * k * sizeof(__fp16);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(__fp16);
    size_t cSize = static_cast<size_t>(m) * n_pes * n * sizeof(__fp16);

    uint8_t *aDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *aHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&aHost), aSize));
    ReadFile(options.GetDataPath("pe_" + std::to_string(pe_id) + "_a.bin"), aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *bHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&bHost), bSize));
    ReadFile(options.GetDataPath("pe_" + std::to_string(pe_id) + "_b.bin"), bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *cHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&cHost), cSize));

    size_t sizeWB = Catcoc::Padding::GetWorkspaceLen(layoutB, L1TileShape::K, L1TileShape::N) * sizeof(ElementB);

    uint8_t *workspaceDevice{nullptr};
    if (isNeedPaddingB) {
        aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST);
    } else {
        workspaceDevice = bDevice;
    }

    void *symmPtr = aclshmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
    uint8_t *gmSymmetric = (uint8_t *)symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling AG_MM kernel " << std::endl;
    uint64_t fftsAddr;
    uint32_t len;
    ACL_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &len));
    for (int i = 0; i < 1; i++) {
        if (isNeedPaddingB) {
            ShmemAllGatherMatmulPadding<true>
                <<<BLOCK_NUM, nullptr, stream>>>(
                fftsAddr,
                aDevice, bDevice, cDevice, workspaceDevice, gmSymmetric,
                m, n, k
            );
        } else {
            ShmemAllGatherMatmulPadding<false>
                <<<BLOCK_NUM, nullptr, stream>>>(
                fftsAddr,
                aDevice, bDevice, cDevice, workspaceDevice, gmSymmetric,
                m, n, k
            );
        }
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling AG_MM kernel " << std::endl;

    ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(options.GetDataPath("aclshmem_output.bin"), cHost, cSize);
    if (pe_id == 0) {
        std::printf("test finished\n");
    }

    aclshmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(cHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(cDevice));
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtFree(workspaceDevice));
    }

    status = aclrtDestroyStream(stream);

    status = aclshmem_finalize();
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in pe " << pe_id << std::endl;
    return 0;
}