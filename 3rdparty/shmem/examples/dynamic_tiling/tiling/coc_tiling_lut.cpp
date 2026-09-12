/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coc_tiling_lut.h"
#include <iostream>
#include <limits>
#include "param.h"

static int GetValueFromMap(int64_t m, int64_t k, int64_t n,
                           const std::map<int, std::vector<std::vector<int>>> &condMap,
                           int defaultVal)
{
    for (const auto &[candidate, condList]: condMap) {
        for (const auto &c: condList) {
            auto in = [&](int64_t v, int lo, int hi) {
                return (lo == -1 || v >= lo) && (hi == -1 || v <= hi);
            };
            if (in(m, c[INDEX0], c[INDEX1]) && in(k, c[INDEX2], c[INDEX3]) && in(n, c[INDEX4], c[INDEX5]))
                return candidate;
        }
    }
    return defaultVal;
}

const std::map<LutKey, const LUTGroup *> g_allLutGroups = {
    {{MATMUL_ALLREDUCE,                    2}, &AllReduce2p},
    {{MATMUL_ALLREDUCE,                    4}, &AllReduce4p},
    {{MATMUL_ALLREDUCE,                    8}, &AllReduce8p},
    {{MATMUL_REDUCE_SCATTER,               2}, &ReduceScatter2p},
    {{MATMUL_REDUCE_SCATTER,               4}, &ReduceScatter4p},
    {{MATMUL_REDUCE_SCATTER,               8}, &ReduceScatter8p},
    {{MATMUL_REDUCE_SCATTER_PADDING,       2}, &ReduceScatter2p},
    {{MATMUL_REDUCE_SCATTER_PADDING,       4}, &ReduceScatter4p},
    {{MATMUL_REDUCE_SCATTER_PADDING,       8}, &ReduceScatter8p},
    {{MATMUL_REDUCE_SCATTER_PADDING_AB,    2}, &ReduceScatter2p},
    {{MATMUL_REDUCE_SCATTER_PADDING_AB,    4}, &ReduceScatter4p},
    {{MATMUL_REDUCE_SCATTER_PADDING_AB,    8}, &ReduceScatter8p},
    {{MATMUL_REDUCE_SCATTER_PADDING_A,     2}, &ReduceScatter2p},
    {{MATMUL_REDUCE_SCATTER_PADDING_A,     4}, &ReduceScatter4p},
    {{MATMUL_REDUCE_SCATTER_PADDING_A,     8}, &ReduceScatter8p},
    {{MATMUL_REDUCE_SCATTER_PADDING_B,     2}, &ReduceScatter2p},
    {{MATMUL_REDUCE_SCATTER_PADDING_B,     4}, &ReduceScatter4p},
    {{MATMUL_REDUCE_SCATTER_PADDING_B,     8}, &ReduceScatter8p},
    {{ALLGATHER_MATMUL,                    2}, &AllGather2p},
    {{ALLGATHER_MATMUL,                    4}, &AllGather4p},
    {{ALLGATHER_MATMUL,                    8}, &AllGather8p},
    {{ALLGATHER_MATMUL_WITH_GATHER_RESULT, 2}, &AllGather2p},
    {{ALLGATHER_MATMUL_WITH_GATHER_RESULT, 4}, &AllGather4p},
    {{ALLGATHER_MATMUL_WITH_GATHER_RESULT, 8}, &AllGather8p},
    {{ALLGATHER_MATMUL_PADDING,            2}, &AllGather2p},
    {{ALLGATHER_MATMUL_PADDING,            4}, &AllGather4p},
    {{ALLGATHER_MATMUL_PADDING,            8}, &AllGather8p},
    // 继续添加...
};

bool ApplyLookupTable(const COCMatMulInfo &info,
                      CocCommType type,
                      int rankSize,
                      CocTilingParams &t)
{
    LutKey key = {type, rankSize};
    auto it = g_allLutGroups.find(key);
    if (it == g_allLutGroups.end()) {
        std::cerr << "[LUT] no table for (" << type << ',' << rankSize << ")\n";
        return false;
    }
    constexpr uint32_t COMM_TILE_M_MULTIPLIER = 2;
    constexpr uint32_t N0_IF_M0_IS_256 = 128;
    constexpr uint32_t N0_IF_M0_IS_NOT_256 = 256;
    constexpr uint32_t DEFAULT_M0 = 256;
    constexpr uint32_t DEFAULT_K0 = 256;
    const LUTGroup &g = *(it->second); // 解引用指针
    auto pick = [&info](auto &mp, int def) { return GetValueFromMap(info.m, info.k, info.n, mp, def); };
    t.m0 = pick(g.m0Map, g.m0Default);
    t.commInterval = pick(g.commIntervalMap, g.commIntervalDefault);
    t.commTileM = pick(g.commTileMMap, g.commTileMDefault) * COMM_TILE_M_MULTIPLIER;
    t.commNpuSplit = pick(g.commNpuSplitMap, g.commNpuSplitDefault);
    t.commDataSplit = pick(g.commDataSplitMap, g.commDataSplitDefault);
    t.commBlockM = t.commTileM;
    t.n0 = (t.m0 == DEFAULT_M0) ? N0_IF_M0_IS_256 : N0_IF_M0_IS_NOT_256;
    t.k0 = DEFAULT_K0;
    return true;
}