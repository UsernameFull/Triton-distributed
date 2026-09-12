/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <map>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <vector>
#include "info.h"
#include "launch_map.h"

struct LUTGroup {
    int m0Default;
    int commIntervalDefault;
    int commTileMDefault;
    int commNpuSplitDefault;
    int commDataSplitDefault;
    std::map<int, std::vector<std::vector<int>>> m0Map;
    std::map<int, std::vector<std::vector<int>>> commIntervalMap;
    std::map<int, std::vector<std::vector<int>>> commTileMMap;
    std::map<int, std::vector<std::vector<int>>> commNpuSplitMap;
    std::map<int, std::vector<std::vector<int>>> commDataSplitMap;
};

using LutKey = std::pair<CocCommType, int>; // {kernelType, rankSize}

bool ApplyLookupTable(const COCMatMulInfo& info, CocCommType type, int rankSize, CocTilingParams& t);

/* ---------- 全局 LUT 声明 ---------- */
extern const LUTGroup AllReduce2p;
extern const LUTGroup AllReduce4p;
extern const LUTGroup AllReduce8p;
extern const LUTGroup ReduceScatter2p;
extern const LUTGroup ReduceScatter4p;
extern const LUTGroup ReduceScatter8p;
extern const LUTGroup AllGather2p;
extern const LUTGroup AllGather4p;
extern const LUTGroup AllGather8p;
extern const std::map<LutKey, const LUTGroup*> g_allLutGroups;