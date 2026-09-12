/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILING_H
#define TILING_H

#include <sstream>
#include <vector>
#include "info.h"
#include "launch_map.h"

std::vector<uint32_t> vCommInterval = {1, 2, 4, 6, 8, 12, 14};
std::vector<uint32_t> vCommTileM = {4, 8, 16, 32, 64};
std::vector<uint32_t> vM0 = {128, 256};
std::vector<std::pair<uint32_t, uint32_t>> vCommSplitNpuDataPair = {{1, 16}, {1, 20}};
std::vector<std::vector<uint32_t>> allParams = {vCommInterval, vCommTileM, vM0};

constexpr uint32_t alignByByte = 512;
constexpr uint32_t alignByElement = alignByByte / sizeof(__fp16);

template <class T>
constexpr T RoundUp(const T &val, const T align)
{
    if (align == 0) {
        return val;
    }
    return (val + align - 1) / align * align;
}

int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

bool IsNeedPadding(uint32_t rows, uint32_t cols, uint32_t trans)
{
    const uint32_t THRESHOLD = 65536;
    if (trans) {
        if (rows < THRESHOLD) {
            return rows % alignByElement != 0;
        } else {
            return true;
        }
    }

    if (cols < THRESHOLD) {
        return cols % alignByElement != 0;
    } else {
        return true;
    }
}

bool CheckCommIntervalReduceScatter(const CocTilingParams &tiling, int rankSize)
{
    constexpr int32_t blockNum = BLOCK_NUM;
    int64_t product = static_cast<int64_t>(blockNum) * tiling.commInterval;

    if (rankSize == 0 || product % rankSize != 0) {
        return false;
    }
    return true;
}

bool CheckCommIntervalAllReduce(const CocTilingParams &tiling, int rankSize)
{
    if (rankSize == 0) {
        return false;
    }

    auto blockCount = MAX_BLOCK_COUNT;
    uint32_t kLoops = CeilDev(tiling.k, tiling.k0);
    int32_t maxPeerMemPerRank = ACLSHMEM_BUFF_BYTES / INPUT_DTYPE / rankSize / blockCount;
    if (tiling.commInterval * tiling.m0 * tiling.k0 * BLOCK_NUM >= maxPeerMemPerRank) {
        return false;
    }
    return true;
}

bool CheckCommIntervalAllGather(const CocTilingParams &tiling, int rankSize)
{
    if (rankSize == 0) {
        return false;
    }

    auto blockCount = MAX_BLOCK_COUNT;
    uint32_t kLoops = CeilDev(tiling.k, tiling.k0);
    int32_t maxPeerMemPerRank = ACLSHMEM_BUFF_BYTES / INPUT_DTYPE / rankSize / blockCount;
    if (tiling.commInterval * tiling.m0 * tiling.k0 * kLoops >= maxPeerMemPerRank) {
        return false;
    }
    return true;
}

void GetParamFromSearchSpace(std::vector<uint32_t>& curParams, std::vector<std::vector<uint32_t>> &results, int pos)
{
    if (pos == allParams.size()) {
        for (int i = 0; i < vCommSplitNpuDataPair.size(); i++) {
            std::vector<uint32_t> tmpParams(curParams.begin(), curParams.end());
            tmpParams.push_back(vCommSplitNpuDataPair[i].first);
            tmpParams.push_back(vCommSplitNpuDataPair[i].second);
            results.push_back(tmpParams);
        }
    } else {
        for (int j = 0; j < allParams[pos].size(); j++) {
            curParams[pos] = allParams[pos][j];
            GetParamFromSearchSpace(curParams, results, pos + 1);
        }
    }
}

void GetTilings(std::vector<CocTilingParams> &tilings, CocTilingParams &t, CocCommType commType, int rankSize)
{
    std::vector<uint32_t> curParams(allParams.size(), 0);
    std::vector<std::vector<uint32_t>> allTilings;
    GetParamFromSearchSpace(curParams, allTilings, 0);
    constexpr uint32_t COMM_TILE_M_MULTIPLIER = 2;
    constexpr uint32_t N0_IF_M0_IS_128 = 256;
    constexpr uint32_t N0_IF_M0_IS_NOT_128 = 128;
    constexpr uint32_t DEFAULT_M0 = 128;
    constexpr uint32_t DEFAULT_K0 = 256;
    for (const auto &tiling : allTilings) {
        uint32_t idx = 0;
        t.commInterval = tiling[idx++];
        t.commTileM    = tiling[idx++] * COMM_TILE_M_MULTIPLIER;
        t.commBlockM   = t.commTileM;
        t.m0           = tiling[idx++];
        t.k0           = DEFAULT_K0;
        t.n0           = (t.m0 == DEFAULT_M0) ? N0_IF_M0_IS_128 : N0_IF_M0_IS_NOT_128;
        t.commNpuSplit = tiling[idx++];
        t.commDataSplit = tiling[idx++];

        if ((commType == ALLGATHER_MATMUL || commType == ALLGATHER_MATMUL_PADDING ||
            commType == ALLGATHER_MATMUL_WITH_GATHER_RESULT)
            && !CheckCommIntervalAllGather(t, rankSize))
            continue;
        if ((commType == MATMUL_REDUCE_SCATTER || commType == MATMUL_REDUCE_SCATTER_PADDING)
            && !CheckCommIntervalReduceScatter(t, rankSize))
            continue;
        if (commType == MATMUL_ALLREDUCE && !CheckCommIntervalAllReduce(t, rankSize))
            continue;
        tilings.push_back(t);
    }
}

bool CreateTilingFile(const std::string filename)
{
    std::ofstream outFile(filename, std::ios::out);
    if (!outFile.is_open()) {
        std::cerr << "Open file failed." << std::endl;
        return false;
    }
    outFile << "Op,M,K,N,Transpose A,Transpose B,M0,commInterval, "
            << "commTileM,commBlockM,commNpuSplit,commDataSplit,Time(us)\n";
    outFile.close();
    return true;
}

bool WriteTilingInfos(std::string opName, std::vector<CocTilingParams> &cocTilings, const std::string filename,
                      int transA = 0, int transB = 1)
{
    std::ofstream outputFile(filename, std::ios::out | std::ios::app);
    if (!outputFile) {
        int err = errno;
        std::error_code ec(err, std::generic_category());
        ERROR_LOG("Open file failed. path = %s, error = %s", filename.c_str(), ec.message().c_str());
        return false;
    }

    for (CocTilingParams cocTiling : cocTilings) {
        outputFile << opName
                   << "," << cocTiling.m
                   << "," << cocTiling.k
                   << "," << cocTiling.n
                   << "," << transA
                   << "," << transB
                   << "," << cocTiling.m0
                   << "," << cocTiling.commInterval
                   << "," << cocTiling.commTileM
                   << "," << cocTiling.commBlockM
                   << "," << cocTiling.commNpuSplit
                   << "," << cocTiling.commDataSplit
                   << "," << "\n";
    }
    outputFile.close();
    return true;
}

size_t GetWorkspaceLen(uint32_t shape0, uint32_t shape1, size_t blockRows, size_t blockCols)
{
    return RoundUp(static_cast<size_t>(shape0), blockRows) *
           RoundUp(static_cast<size_t>(shape1), blockCols);
}

#endif // TILING_H