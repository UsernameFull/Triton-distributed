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

constexpr int32_t REDUCESCATTER_910B_TWO_RANK_FP16_M0_DEFAULT = 128;
constexpr int32_t REDUCESCATTER_910B_TWO_RANK_FP16_COMMDATASPLIT_DEFAULT = 20;
constexpr int32_t REDUCESCATTER_910B_TWO_RANK_FP16_COMMNPUSPLIT_DEFAULT = 1;
constexpr int32_t REDUCESCATTER_910B_TWO_RANK_FP16_COMMINTERVAL_DEFAULT = 8;
constexpr int32_t REDUCESCATTER_910B_TWO_RANK_FP16_COMMTILEM_DEFAULT = 4;
constexpr int32_t REDUCESCATTER_910B_FOUR_RANK_FP16_COMMNPUSPLIT_DEFAULT = 4;
constexpr int32_t REDUCESCATTER_910B_FOUR_RANK_FP16_COMMTILEM_DEFAULT = 32;
constexpr int32_t REDUCESCATTER_910B_FOUR_RANK_FP16_COMMINTERVAL_DEFAULT = 8;
constexpr int32_t REDUCESCATTER_910B_FOUR_RANK_FP16_COMMDATASPLIT_DEFAULT = 4;
constexpr int32_t REDUCESCATTER_910B_FOUR_RANK_FP16_M0_DEFAULT = 128;
constexpr int32_t REDUCESCATTER_910B_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT = 16;
constexpr int32_t REDUCESCATTER_910B_EIGHT_RANK_FP16_COMMNPUSPLIT_DEFAULT = 1;
constexpr int32_t REDUCESCATTER_910B_EIGHT_RANK_FP16_COMMINTERVAL_DEFAULT = 14;
constexpr int32_t REDUCESCATTER_910B_EIGHT_RANK_FP16_COMMTILEM_DEFAULT = 32;
constexpr int32_t REDUCESCATTER_910B_EIGHT_RANK_FP16_M0_DEFAULT = 128;

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BTwoRankFP16CommtilemMap = {
    {4.0,
        {{-1, 2147483647, -1, 2147483647, -1, 72},  {-1, 4608, -1, 18432, 832, 2352}}},
    {2.0,
        {{-1, 4608, -1, 2147483647, 72, 832}, {-1, 4608, 18432, 2147483647, 832, 2352},
            {-1, 4608, -1, 2147483647, 2352, 2147483647}, {4608, 2147483647, -1, 2147483647, 72, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BTwoRankFP16CommintervalMap = {
    {6,
        {{-1, 13312, -1, 2147483647, -1, 192}, {13312, 2147483647, -1, 2147483647, 72, 192},
            {17152, 20224, -1, 4608, 192, 2147483647}, {23084, 2147483647, 2224, 4608, 192, 2147483647}}},
    {5,
        {{13312, 22528, -1, 2147483647, -1, 72}, {-1, 14336, -1, 7360, 13312, 2147483647},
            {14336, 17152, -1, 4608, 192, 6656}}},
    {8,
        {{22528, 2147483647, -1, 2147483647, -1, 72}, {14336, 17152, -1, 4608, 6656, 2147483647}}},
    {1,
        {{-1, 14336, -1, 7360, 192, 1152}, {-1, 2992, 7360, 19456, 192, 2147483647},
            {-1, 2992, 30720, 2147483647, 192, 2147483647}, {2992, 17152, 7360, 2147483647, 192, 2147483647},
            {17152, 81920, 6656, 10240, 192, 2147483647}, {17152, 2147483647, 10240, 2147483647, 192, 2147483647}}},
    {4,
        {{-1, 14336, -1, 7360, 1152, 13312}, {14336, 17152, 4608, 7360, 192, 2147483647},
            {-1, 2992, 19456, 30720, 192, 2147483647}, {20224, 23084, -1, 4608, 192, 2147483647},
            {23084, 2147483647, -1, 2224, 192, 2147483647}, {17152, 81920, 4608, 6656, 192, 2147483647},
            {81920, 2147483647, 4608, 10240, 192, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BTwoRankFP16CommnpusplitMap = {
    {2,
        {{-1, 2992, -1, 2147483647, -1, 641},  {-1, 2992, -1, 2147483647, 2432, 14016},
            {2992, 2147483647, 3968, 4608, -1, 1792}, {3504, 2147483647, 6784, 2147483647, -1, 14016}}},
    {1,
        {{-1, 2992, -1, 2147483647, 641, 2432}, {2992, 2147483647, -1, 3968, -1, 14016},
            {2992, 2147483647, 4608, 6784, -1, 1792}, {2992, 2147483647, 3968, 6784, 1792, 14016},
            {2992, 3504, 6784, 2147483647, -1, 14016}, {-1, 2147483647, -1, 2147483647, 14016, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BTwoRankFP16CommdatasplitMap = {
    {8,
        {{-1, 2992, -1, 2147483647, -1, 641},  {-1, 2992, -1, 2147483647, 2432, 14016},
            {2992, 2147483647, 3968, 4608, -1, 1792}, {3504, 2147483647, 6784, 2147483647, -1, 14016}}},
    {20,
        {{-1, 2992, -1, 2147483647, 641, 2432}, {2992, 2147483647, -1, 3968, -1, 14016},
            {2992, 2147483647, 4608, 6784, -1, 1792}, {2992, 2147483647, 3968, 6784, 1792, 14016},
            {2992, 3504, 6784, 2147483647, -1, 14016}, {-1, 2147483647, -1, 2147483647, 14016, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BTwoRankFP16M0Map = {
    {256,
        {{-1, 2147483647, -1, 2147483647, -1,  192}, {-1, 5292, 4608, 2147483647, 448, 3632},
            {-1, 5292, 38656, 2147483647, 3632, 2147483647}}},
    {128,
        {{-1, 5292, -1, 2147483647, 192, 448}, {-1, 5292, -1, 4608, 448, 3632},
            {-1, 5292, -1, 38656, 3632, 2147483647}, {5292, 2147483647, -1, 2147483647, 192, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BFourRankFP16M0Map = {
    {128,
        {{-1, 13312, -1, 2147483647, -1, 448}, {13312, 2147483647, 576, 2147483647, 72, 192},
            {-1, 2147483647, -1, 38912, 448, 2147483647}}},
    {256,
        {{13312, 2147483647, -1, 2147483647, -1, 72},  {13312, 2147483647, -1, 576, 72, 192},
            {13312, 2147483647, -1, 2147483647, 192, 448}, {-1, 2147483647, 38912, 2147483647, 448, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BFourRankFP16CommdatasplitMap = {
    {4,
        {{-1, 164096, -1, 2147483647, -1, 2816}, {164096, 2147483647, -1, 2147483647, -1, 1344},
            {-1, 2147483647, -1, 14000, 2816, 5760}, {-1, 2147483647, 29696, 2147483647, 2816, 5760},
            {-1, 2147483647, -1, 2147483647, 5760, 2147483647}}},
    {20,
        {{164096, 2147483647, -1, 2147483647, 1344, 2816}}},
    {16,
        {{-1, 2147483647, 14000, 29696, 2816, 5760}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BFourRankFP16CommintervalMap = {
    {1,
        {{-1, 28624, -1, 402, -1, 1616}, {-1, 28624, 2816, 2147483647, -1, 1616},
            {-1, 16360, 5888, 50048, 1744, 8064}, {16360, 28624, 5888, 2147483647, 1744, 8064},
            {28624, 2147483647, 576, 2147483647, 384, 3584},  {-1, 2147483647, 5824, 2147483647, 8064, 2147483647}}},
    {2,
        {{-1, 28624, 402, 2816, -1, 1616}, {-1, 28624, -1, 2147483647, 1616, 1744},
            {-1, 6132, -1, 5888, 1744, 8064}, {16360, 28624, -1, 5888, 4608, 8064}}},
    {4,
        {{6132, 16360, -1, 5888, 1744, 8064}, {16360, 28624, -1, 5888, 1744, 4608},
            {28624, 2147483647, 576, 2147483647, -1, 384},  {28624, 2147483647, -1, 5888, 3584, 4608},
            {-1, 16360, -1, 4800, 8064, 11648}, {-1, 16360, -1, 4800, 16128, 2147483647}}},
    {6,
        {{-1, 16360, 50048, 2147483647, 1744, 8064}, {28624, 2147483647, 102, 576, -1, 3584},
            {28624, 2147483647, 5888, 2147483647, 3584, 4608}, {28624, 2147483647, -1, 2147483647, 4608, 8064},
            {-1, 16360, -1, 4800, 11648, 16128}, {16360, 2147483647, -1, 4800, 8064, 2147483647},
            {-1, 2147483647, 4800, 5824, 8064, 2147483647}}},
    {8,
        {{28624, 2147483647, -1, 102, -1, 3584}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BFourRankFP16CommtilemMap = {
    {16.0,
        {{-1, 81920, -1, 2147483647, -1, 72}}},
    {8.0,
        {{-1, 81920, -1, 2147483647, 72, 192}, {81920, 2147483647, -1, 2147483647, -1, 192},
            {-1, 2147483647, -1, 722, 192, 640}, {-1, 2147483647, 30720, 2147483647, 192, 1472}}},
    {4.0,
        {{-1, 2147483647, -1, 722, 640, 1472}, {-1, 2147483647, 722, 30720, 192, 1472},
            {-1, 4352, -1, 2147483647, 1472, 2147483647}, {4864, 2147483647, -1, 2147483647, 1472, 2147483647}}},
    {32.0,
        {{4352, 4864, -1, 2147483647, 1472, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BFourRankFP16CommnpusplitMap = {
    {4,
        {{-1, 164096, -1, 2147483647, -1, 2816}, {164096, 2147483647, -1, 2147483647, -1, 1344},
            {-1, 2147483647, -1, 14000, 2816, 5760}, {-1, 2147483647, 29696, 2147483647, 2816, 5760},
            {-1, 2147483647, -1, 2147483647, 5760, 2147483647}}},
    {1,
        {{164096, 2147483647, -1, 2147483647, 1344, 2816}, {-1, 2147483647, 14000, 29696, 2816, 5760}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BEightRankFP16M0Map = {
    {256,
        {{-1, 22528, -1, 2147483647, -1, 192}, {22528, 2147483647, -1, 2147483647, 72, 192},
            {-1, 2147483647, -1, 16896, 192, 832}, {-1, 2147483647, 38912, 2147483647, 832, 2147483647}}},
    {128,
        {{22528, 2147483647, -1, 2147483647, -1, 72},  {-1, 2147483647, 16896, 2147483647, 192, 832},
            {-1, 2147483647, -1, 38912, 832, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BEightRankFP16CommtilemMap = {
    {32.0,
        {{-1, 19456, -1, 2147483647, -1, 832},  {19456, 2147483647, -1, 2147483647, -1, 192},
            {-1, 2147483647, 29824, 2147483647, 832, 8064}}},
    {8.0,
        {{19456, 2147483647, -1, 2147483647, 192, 832},  {-1, 2147483647, -1, 22656, 832, 8064},
            {-1, 8960, -1, 2147483647, 8064, 2147483647}, {11520, 2147483647, -1, 2147483647, 8064, 2147483647}}},
    {16.0,
        {{-1, 2147483647, 22656, 29824, 832, 8064}, {8960,  11520, -1, 2147483647, 8064, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BEightRankFP16CommintervalMap = {
    {6,
        {{-1, 13312, -1, 2147483647, -1, 192}, {13312, 2147483647, -1, 2147483647, 72, 192},
            {1536, 4608, 31104, 2147483647, 4352, 14016}, {1536, 4608, -1, 2147483647, 14016, 2147483647},
            {4608, 27648, 12288, 2147483647, 192, 1152}, {27648, 2147483647, -1, 2147483647, 192,  1152},
            {4608, 2147483647, -1, 1878, 1152, 2147483647}}},
    {4,
        {{13312, 22528, -1, 2147483647, -1, 72},  {-1, 1536, -1, 20480, 192, 2147483647},
            {1536, 4608, -1, 2147483647, 192, 4352}, {1536, 4608, -1, 31104, 4352, 14016},
            {4608, 27648, -1, 12288, 192, 1152}, {4608, 2147483647, 1878, 20608, 1152, 2147483647}}},
    {12,
        {{22528, 2147483647, -1, 2147483647, -1, 72}}},
    {8,
        {{-1, 1536, 20480, 2147483647, 192, 2147483647}}},
    {14,
        {{4608, 2147483647, 20608, 2147483647, 1152, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BEightRankFP16CommnpusplitMap = {
    {8,
        {{-1, 7168, -1, 2147483647, -1, 832}, {7168, 2147483647, -1, 2147483647, 72, 192},
            {-1, 2147483647, 8000, 9088, 3248, 3760}}},
    {1,
        {{7168, 2147483647, -1, 2147483647, -1, 72},  {7168, 2147483647, -1, 2147483647, 192, 832},
            {-1, 2147483647, -1, 8000, 832, 2147483647}, {-1, 2147483647, 8000, 9088, 832, 3248},
            {-1, 2147483647, 8000, 9088, 3760, 2147483647}, {-1, 2147483647, 9088, 2147483647, 832, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_reducescatter910BEightRankFP16CommdatasplitMap = {
    {2,
        {{-1, 7168, -1, 2147483647, -1, 832}, {7168, 2147483647, -1, 2147483647, 72, 192},
            {-1, 2147483647, 8000, 9088, 3248, 3760}}},
    {20,
        {{7168, 13312, -1, 2147483647, -1, 72},  {3504, 10752, 8000, 9088, 832, 2176}}},
    {16,
        {{13312, 2147483647, -1, 2147483647, -1, 72},  {7168, 273408, -1, 2147483647, 72, 832},
            {-1, 2147483647, -1, 8000, 832, 2147483647}, {-1, 3504, 8000, 2147483647, 832, 2224},
            {-1, 3504, 8000, 2147483647, 3632, 2147483647}, {10752, 2147483647, 8000, 9088, 832, 2176},
            {3504, 2147483647, 8000, 9088, 2176, 2147483647}, {3504, 2147483647, 9088, 2147483647, 832, 2147483647}}}
};

const LUTGroup ReduceScatter2p{
    REDUCESCATTER_910B_TWO_RANK_FP16_M0_DEFAULT,
    REDUCESCATTER_910B_TWO_RANK_FP16_COMMINTERVAL_DEFAULT,
    REDUCESCATTER_910B_TWO_RANK_FP16_COMMTILEM_DEFAULT,
    REDUCESCATTER_910B_TWO_RANK_FP16_COMMNPUSPLIT_DEFAULT,
    REDUCESCATTER_910B_TWO_RANK_FP16_COMMDATASPLIT_DEFAULT,
    g_reducescatter910BTwoRankFP16M0Map,
    g_reducescatter910BTwoRankFP16CommintervalMap,
    g_reducescatter910BTwoRankFP16CommtilemMap,
    g_reducescatter910BTwoRankFP16CommnpusplitMap,
    g_reducescatter910BTwoRankFP16CommdatasplitMap
};

const LUTGroup ReduceScatter4p{
    REDUCESCATTER_910B_FOUR_RANK_FP16_M0_DEFAULT,
    REDUCESCATTER_910B_FOUR_RANK_FP16_COMMINTERVAL_DEFAULT,
    REDUCESCATTER_910B_FOUR_RANK_FP16_COMMTILEM_DEFAULT,
    REDUCESCATTER_910B_FOUR_RANK_FP16_COMMNPUSPLIT_DEFAULT,
    REDUCESCATTER_910B_FOUR_RANK_FP16_COMMDATASPLIT_DEFAULT,
    g_reducescatter910BFourRankFP16M0Map,
    g_reducescatter910BFourRankFP16CommintervalMap,
    g_reducescatter910BFourRankFP16CommtilemMap,
    g_reducescatter910BFourRankFP16CommnpusplitMap,
    g_reducescatter910BFourRankFP16CommdatasplitMap
};

const LUTGroup ReduceScatter8p{
    REDUCESCATTER_910B_EIGHT_RANK_FP16_M0_DEFAULT,
    REDUCESCATTER_910B_EIGHT_RANK_FP16_COMMINTERVAL_DEFAULT,
    REDUCESCATTER_910B_EIGHT_RANK_FP16_COMMTILEM_DEFAULT,
    REDUCESCATTER_910B_EIGHT_RANK_FP16_COMMNPUSPLIT_DEFAULT,
    REDUCESCATTER_910B_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT,
    g_reducescatter910BEightRankFP16M0Map,
    g_reducescatter910BEightRankFP16CommintervalMap,
    g_reducescatter910BEightRankFP16CommtilemMap,
    g_reducescatter910BEightRankFP16CommnpusplitMap,
    g_reducescatter910BEightRankFP16CommdatasplitMap
};