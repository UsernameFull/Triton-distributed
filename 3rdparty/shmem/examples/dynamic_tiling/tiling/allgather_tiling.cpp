/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <map>
#include <vector>
#include "coc_tiling_lut.h"

constexpr int32_t ALLGATHER_910B_TWO_RANK_FP16_COMMINTERVAL_DEFAULT = 14;
constexpr int32_t ALLGATHER_910B_TWO_RANK_FP16_M0_DEFAULT = 128;
constexpr int32_t ALLGATHER_910B_TWO_RANK_FP16_COMMNPUSPLIT_DEFAULT = 2;
constexpr int32_t ALLGATHER_910B_TWO_RANK_FP16_COMMTILEM_DEFAULT = 32;
constexpr int32_t ALLGATHER_910B_TWO_RANK_FP16_COMMDATASPLIT_DEFAULT = 20;
constexpr int32_t ALLGATHER_910B_FOUR_RANK_FP16_COMMINTERVAL_DEFAULT = 14;
constexpr int32_t ALLGATHER_910B_FOUR_RANK_FP16_M0_DEFAULT = 128;
constexpr int32_t ALLGATHER_910B_FOUR_RANK_FP16_COMMDATASPLIT_DEFAULT = 20;
constexpr int32_t ALLGATHER_910B_FOUR_RANK_FP16_COMMTILEM_DEFAULT = 32;
constexpr int32_t ALLGATHER_910B_FOUR_RANK_FP16_COMMNPUSPLIT_DEFAULT = 4;
constexpr int32_t ALLGATHER_910B_EIGHT_RANK_FP16_COMMNPUSPLIT_DEFAULT = 1;
constexpr int32_t ALLGATHER_910B_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT = 20;
constexpr int32_t ALLGATHER_910B_EIGHT_RANK_FP16_COMMTILEM_DEFAULT = 32;
constexpr int32_t ALLGATHER_910B_EIGHT_RANK_FP16_M0_DEFAULT = 128;
constexpr int32_t ALLGATHER_910B_EIGHT_RANK_FP16_COMMINTERVAL_DEFAULT = 14;

static std::map<int, std::vector<std::vector<int>>> g_allgather910BTwoRankFP16CommdatasplitMap = {
    {8,
        {{-1, 6656, -1, 2147483647, -1, 72}, {6656, 2147483647, 64, 2147483647, -1, 72},
            {-1, 2147483647, -1, 2147483647, 72, 2147483647}}},
    {20,
        {{-1, 2147483647, -1, 64, -1, 72}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BTwoRankFP16CommtilemMap = {
    {32.0,
        {{-1, 8180, -1, 2147483647, -1, 1152}, {-1, 8180, 1408, 2147483647, 1152, 1344},
            {-1, 8180, -1, 2147483647, 1344, 8064}, {8180, 2147483647, 64, 2147483647, -1, 3584},
            {8180, 2147483647, 1712, 2147483647, 3584, 8064}, {4864, 10006, 10560, 2147483647, 8064, 10496},
            {1152, 10006, -1, 2147483647, 10496, 14016}, {1152, 10006, -1, 2147483647, 16128, 2147483647}}},
    {16.0,
        {{-1, 8180, -1, 1408, 1152, 1344}, {-1, 1152, -1, 2147483647, 8064, 2147483647},
            {1152, 4864, -1, 2147483647, 8064, 10496}, {4864, 10006, -1, 10560, 8064, 10496},
            {1152, 10006, -1, 2147483647, 14016, 16128}, {10006, 2147483647, -1, 2147483647, 8064, 2147483647}}},
    {4.0,
        {{8180, 2147483647, -1, 64, -1, 8064}}},
    {8.0,
        {{8180, 2147483647, 64, 1712, 3584, 8064}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BTwoRankFP16CommnpusplitMap = {
    {2,
        {{-1, 6656, -1, 2147483647, -1, 72}, {6656, 2147483647, 64, 2147483647, -1, 72},
            {-1, 2147483647, -1, 2147483647, 72, 2147483647}}},
    {1,
        {{6656, 2147483647, -1, 64, -1, 72}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BTwoRankFP16M0Map = {
    {256,
        {{-1, 39936, -1, 2147483647, -1, 192}, {39936, 2147483647, -1, 576, -1, 192},
            {-1, 3200, 38912, 2147483647, 4352, 2147483647}, {11542, 2147483647, -1, 2147483647, 7424, 2147483647}}},
    {128,
        {{39936, 2147483647, 576, 2147483647, -1, 192}, {-1, 3200, -1, 38912, 192, 2147483647},
            {-1, 3200, 38912, 2147483647, 192, 4352}, {3200, 11542, -1, 2147483647, 192, 2147483647},
            {11542, 2147483647, -1, 2147483647, 192, 7424}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BTwoRankFP16CommintervalMap = {
    {8,
        {{-1, 1920, -1, 2147483647, -1, 448}, {4608, 2147483647, -1, 2147483647, 20224, 2147483647}}},
    {14,
        {{-1, 1920, -1, 2147483647, 448, 832}, {1920, 2147483647, -1, 2147483647, -1, 1152},
            {1920, 2147483647, 7168, 2147483647, 1152, 1616}, {4608, 2147483647, -1, 6656, 1616, 3584},
            {40960, 2147483647, 6656, 2147483647, 1616, 3584}}},
    {12,
        {{-1, 1920, -1, 7168, 832, 1616}, {1920, 5268, -1, 7168, 1152, 1616},
            {4608, 40960, 6656, 2147483647, 1616, 3584}, {12264, 2147483647, -1, 2147483647, 3584, 4608}}},
    {6,
        {{-1, 1920, 7168, 21504, 832, 1616}, {4608, 12264, -1, 2147483647, 3584, 4608},
            {4608, 2147483647, 3584, 2147483647, 4608, 5888}, {4608, 2147483647, -1, 2147483647, 12672, 20224}}},
    {1,
        {{-1, 1920, 21504, 2147483647, 832, 1616}, {-1, 1152, -1, 2147483647, 1744, 2147483647},
            {1152, 4608, 1408, 2147483647, 1744, 11648}, {4608, 2147483647, -1, 7040, 9216, 12672}}},
    {4,
        {{5268, 2147483647, -1, 7168, 1152, 1616}, {-1, 4608, -1, 2147483647, 1616, 1744},
            {1152, 4608, -1, 1408, 1744, 2147483647}, {1152, 4608, 1408, 2147483647, 11648, 2147483647},
            {4608, 2147483647, -1, 3584, 4608, 5888}, {4608, 2147483647, -1, 2147483647, 5888, 9216},
            {4608, 2147483647, 7040, 2147483647, 9216, 12672}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BFourRankFP16CommnpusplitMap = {
    {4,
        {{-1, 3328, -1, 2147483647, -1, 72}, {3328, 2147483647, 2048, 2147483647, -1, 72},
            {-1, 960, -1, 38912, 72, 2147483647}, {960, 41024, -1, 2147483647, 72, 2147483647},
            {90176, 2147483647, -1, 2147483647, 72, 2147483647}}},
    {1,
        {{3328, 2147483647, -1, 2048, -1, 72}, {-1, 960, 38912, 2147483647, 72, 2147483647},
            {41024, 90176, -1, 2147483647, 72, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BFourRankFP16CommtilemMap = {
    {32.0,
        {{-1, 90176, -1, 64, -1, 192}, {-1, 90176, 576, 2147483647, -1, 72},
            {-1, 90176, 14674, 2147483647, 192, 448}, {-1, 704, -1, 2147483647, 448, 3248},
            {704, 960, -1, 2147483647, 448, 1152}, {960, 90176, -1, 2147483647, 448, 3248},
            {-1, 7156, -1, 3712, 13696, 2147483647}, {-1, 7156, 3712, 25088, 3248, 2147483647},
            {-1, 7156, 25088, 2147483647, 3248, 6272}, {7156, 2147483647, 2992, 3504, 3248, 2147483647},
            {7156, 2147483647, 4544, 2147483647, 3248, 2147483647}}},
    {16.0,
        {{-1, 90176, 64, 576, -1, 192}, {90176, 2147483647, -1, 2147483647, -1, 3248},
            {-1, 7156, -1, 3712, 3248, 13696}, {-1, 7156, 25088, 2147483647, 6272, 2147483647},
            {7156, 2147483647, -1, 2992, 3248, 2147483647}, {7156, 2147483647, 3504, 4544, 3248, 2147483647}}},
    {8.0,
        {{-1, 90176, 576, 2147483647, 72, 192}, {-1, 90176, -1, 14674, 192, 448}}},
    {4.0,
        {{704, 960, -1, 2147483647, 1152, 3248}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BFourRankFP16CommdatasplitMap = {
    {4,
        {{-1, 3328, -1, 2147483647, -1, 72}, {3328, 2147483647, 2048, 2147483647, -1, 72},
            {-1, 960, -1, 38912, 72, 2147483647}, {960, 41024, -1, 2147483647, 72, 2147483647},
            {90176, 2147483647, -1, 2147483647, 72, 2147483647}}},
    {20,
        {{3328, 2147483647, -1, 2048, -1, 72}, {41024, 90176, -1, 2147483647, 72, 2147483647}}},
    {16,
        {{-1, 960, 38912, 2147483647, 72, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BFourRankFP16M0Map = {
    {128,
        {{-1, 960, -1, 2147483647, -1, 448}, {-1, 960, -1, 2147483647, 832, 3248},
            {960, 2147483647, 402, 850, 192, 3248}, {-1, 576, -1, 2147483647, 3248, 2147483647},
            {576, 876, -1, 2147483647, 3248, 4800}, {576, 3904, -1, 9088, 4800, 2147483647},
            {3904, 2147483647, -1, 2147483647, 3248, 4608}, {3904, 2147483647, -1, 2134, 4608, 2147483647}}},
    {256,
        {{-1, 960, -1, 2147483647, 448, 832}, {960, 2147483647, -1, 850, -1, 192},
            {960, 2147483647, -1, 402, 192, 3248}, {960, 2147483647, 850, 2147483647, -1, 3248},
            {876, 3904, -1, 2147483647, 3248, 4800}, {576, 3904, 9088, 2147483647, 4800, 2147483647},
            {3904, 2147483647, 2134, 2147483647, 4608, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BFourRankFP16CommintervalMap = {
    {1,
        {{-1, 1600, -1, 2147483647, -1, 448}, {1600, 2147483647, 2560, 2147483647, -1, 448},
            {1600, 2147483647, -1, 2147483647, 448, 832}, {-1, 1312, 896, 2147483647, 832, 29824},
            {1355, 4078, 896, 16768, 832, 2147483647}, {4078, 2147483647, 896, 2147483647, 832, 3584},
            {4078, 2147483647, 10240, 2147483647, 3584, 4608}, {4078, 2147483647, 1366, 2147483647, 4608, 2147483647}}},
    {14,
        {{-1, 1600, -1, 2147483647, 448, 832}, {1600, 2147483647, -1, 2560, -1, 448},
            {-1, 1312, 896, 2147483647, 29824, 2147483647}}},
    {12,
        {{-1, 2147483647, -1, 448, 832, 1920}, {-1, 2147483647, -1, 896, 1920, 2147483647}}},
    {6,
        {{-1, 2147483647, 448, 896, 832, 1920}}},
    {4,
        {{1312, 1355, 896, 2147483647, 832, 2147483647}, {1355, 4078, 16768, 2147483647, 832, 2147483647},
            {4078, 2147483647, 896, 10240, 3584, 4608}, {4078, 2147483647, 896, 1366, 4608, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BEightRankFP16CommintervalMap = {
    {8,
        {{-1, 2816, -1, 2147483647, -1, 72}, {-1, 10240, -1, 2147483647, 72, 192}}},
    {14,
        {{2816, 3840, -1, 2147483647, -1, 72}}},
    {1,
        {{3840, 10240, -1, 2147483647, -1, 72}, {-1, 3072, -1, 1152, 192, 2147483647},
            {-1, 2147483647, 1152, 2147483647, 192, 34304}}},
    {12,
        {{10240, 2147483647, -1, 2147483647, -1, 192}}},
    {6,
        {{3072, 2147483647, -1, 1152, 192, 2147483647}}},
    {4,
        {{-1, 2147483647, 1152, 2147483647, 34304, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BEightRankFP16M0Map = {
    {256,
        {{-1, 3840, -1, 2147483647, -1, 192}, {3840, 10240, -1, 2147483647, 72, 192},
            {40960, 2147483647, -1, 2147483647, 72, 192}, {-1, 2039, -1, 38912, 448, 832},
            {-1, 2039, -1, 896, 832, 2147483647}, {2039, 2147483647, -1, 38912, 19840, 2147483647},
            {-1, 2147483647, 38912, 2147483647, 192, 2147483647}}},
    {128,
        {{3840, 2147483647, -1, 2147483647, -1, 72}, {10240, 40960, -1, 2147483647, 72, 192},
            {-1, 2039, -1, 38912, 192, 448}, {-1, 2039, 896, 38912, 832, 2147483647},
            {2039, 2147483647, -1, 38912, 192, 19840}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BEightRankFP16CommtilemMap = {
    {8.0,
        {{-1, 2816, -1, 2147483647, -1, 192}, {2816, 40960, 576, 2147483647, 72, 192},
            {2039, 2147483647, -1, 3072, 192, 3584}}},
    {16.0,
        {{2816, 3840, -1, 2147483647, -1, 72}, {2816, 40960, -1, 576, 72, 192},
            {40960, 2147483647, -1, 2147483647, 72, 192}, {2501, 2147483647, -1, 14336, 3584, 4608},
            {2039, 2501, -1, 2560, 4608, 2147483647}}},
    {32.0,
        {{3840, 2147483647, -1, 2147483647, -1, 72}, {-1, 2039, -1, 2147483647, 192, 2147483647},
            {2039, 2147483647, 3072, 2147483647, 192, 3584}, {2039, 2501, -1, 14336, 3584, 4608},
            {2039, 2501, 2560, 2147483647, 4608, 2147483647}, {2501, 2147483647, -1, 2147483647, 4608, 2147483647}}},
    {4.0,
        {{2039, 2147483647, 14336, 2147483647, 3584, 4608}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BEightRankFP16CommdatasplitMap = {
    {20,
        {{-1, 2816, -1, 2147483647, -1, 192}}},
    {16,
        {{2816, 3840, -1, 2147483647, -1, 72}, {2816, 2147483647, -1, 2147483647, 72, 192},
            {2501, 2147483647, 14336, 2147483647, 192, 2147483647}}},
    {2,
        {{3840, 2147483647, -1, 2147483647, -1, 72}, {-1, 2501, -1, 2147483647, 192, 2147483647},
            {2501, 2147483647, -1, 14336, 192, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allgather910BEightRankFP16CommnpusplitMap = {
    {1,
        {{-1, 3840, -1, 2147483647, -1, 192}, {2816, 3840, -1, 2147483647, -1, 72},
            {2816, 2147483647, -1, 2147483647, 72, 192}, {2501, 2147483647, 14336, 2147483647, 192, 2147483647}}},
    {8,
        {{3840, 2147483647, -1, 2147483647, -1, 72}, {-1, 2501, -1, 2147483647, 192, 2147483647},
            {2501, 2147483647, -1, 14336, 192, 2147483647}}}
};

const LUTGroup AllGather2p{
    ALLGATHER_910B_TWO_RANK_FP16_M0_DEFAULT,
    ALLGATHER_910B_TWO_RANK_FP16_COMMINTERVAL_DEFAULT,
    ALLGATHER_910B_TWO_RANK_FP16_COMMTILEM_DEFAULT,
    ALLGATHER_910B_TWO_RANK_FP16_COMMNPUSPLIT_DEFAULT,
    ALLGATHER_910B_TWO_RANK_FP16_COMMDATASPLIT_DEFAULT,
    g_allgather910BTwoRankFP16M0Map,
    g_allgather910BTwoRankFP16CommintervalMap,
    g_allgather910BTwoRankFP16CommtilemMap,
    g_allgather910BTwoRankFP16CommnpusplitMap,
    g_allgather910BTwoRankFP16CommdatasplitMap
};

const LUTGroup AllGather4p{
    ALLGATHER_910B_FOUR_RANK_FP16_M0_DEFAULT,
    ALLGATHER_910B_FOUR_RANK_FP16_COMMINTERVAL_DEFAULT,
    ALLGATHER_910B_FOUR_RANK_FP16_COMMTILEM_DEFAULT,
    ALLGATHER_910B_FOUR_RANK_FP16_COMMNPUSPLIT_DEFAULT,
    ALLGATHER_910B_FOUR_RANK_FP16_COMMDATASPLIT_DEFAULT,
    g_allgather910BFourRankFP16M0Map,
    g_allgather910BFourRankFP16CommintervalMap,
    g_allgather910BFourRankFP16CommtilemMap,
    g_allgather910BFourRankFP16CommnpusplitMap,
    g_allgather910BFourRankFP16CommdatasplitMap
};

const LUTGroup AllGather8p{
    ALLGATHER_910B_EIGHT_RANK_FP16_M0_DEFAULT,
    ALLGATHER_910B_EIGHT_RANK_FP16_COMMINTERVAL_DEFAULT,
    ALLGATHER_910B_EIGHT_RANK_FP16_COMMTILEM_DEFAULT,
    ALLGATHER_910B_EIGHT_RANK_FP16_COMMNPUSPLIT_DEFAULT,
    ALLGATHER_910B_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT,
    g_allgather910BEightRankFP16M0Map,
    g_allgather910BEightRankFP16CommintervalMap,
    g_allgather910BEightRankFP16CommtilemMap,
    g_allgather910BEightRankFP16CommnpusplitMap,
    g_allgather910BEightRankFP16CommdatasplitMap
};