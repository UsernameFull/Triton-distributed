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

constexpr int32_t ALLREDUCE_910B_TWO_RANK_FP16_M0_DEFAULT = 128;
constexpr int32_t ALLREDUCE_910B_TWO_RANK_FP16_COMMNPUSPLIT_DEFAULT = 2;
constexpr int32_t ALLREDUCE_910B_TWO_RANK_FP16_PVALUE_DEFAULT = 8;
constexpr int32_t ALLREDUCE_910B_TWO_RANK_FP16_UBMOVENUM_DEFAULT = 32;
constexpr int32_t ALLREDUCE_910B_TWO_RANK_FP16_COMMDATASPLIT_DEFAULT = 8;
constexpr int32_t ALLREDUCE_910B_FOUR_RANK_FP16_M0_DEFAULT = 128;
constexpr int32_t ALLREDUCE_910B_FOUR_RANK_FP16_COMMNPUSPLIT_DEFAULT = 4;
constexpr int32_t ALLREDUCE_910B_FOUR_RANK_FP16_COMMDATASPLIT_DEFAULT = 4;
constexpr int32_t ALLREDUCE_910B_FOUR_RANK_FP16_PVALUE_DEFAULT = 6;
constexpr int32_t ALLREDUCE_910B_FOUR_RANK_FP16_UBMOVENUM_DEFAULT = 8;
constexpr int32_t ALLREDUCE_910B_EIGHT_RANK_FP16_M0_DEFAULT = 128;
constexpr int32_t ALLREDUCE_910B_EIGHT_RANK_FP16_COMMNPUSPLIT_DEFAULT = 1;
constexpr int32_t ALLREDUCE_910B_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT = 20;
constexpr int32_t ALLREDUCE_910B_EIGHT_RANK_FP16_PVALUE_DEFAULT = 14;
constexpr int32_t ALLREDUCE_910B_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT = 32;

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BTwoRankFP16CommdatasplitMap = {
    {8,
     {{-1, 2147483647, -1, 2147483647, -1, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BTwoRankFP16UbmovenumMap = {
    {32.0,
        {{-1, 3328, -1, 2147483647, -1, 832}, {3328, 2147483647, -1, 2112, -1, 72},
            {3328, 2147483647, -1, 2147483647, 72, 192}, {3328, 18432, 2898, 2147483647, 576, 832},
            {-1, 2147483647, 2816, 3840, 832, 1472}}},
    {16.0,
        {{3328, 2147483647, 2112, 2147483647, -1, 72}, {3328, 2147483647, -1, 2898, 192, 832},
            {3328, 18868, 2898, 2147483647, 192, 576}, {18432, 2147483647, 2898, 2147483647, 576, 832},
            {-1, 2147483647, 5632, 7168, 832, 1472}, {1536, 3504, -1, 2048, 1472, 2147483647},
            {3504, 3840, -1, 2147483647, 1472, 2147483647}}},
    {8.0,
        {{18868, 2147483647, 2898, 2147483647, 192, 576}, {-1, 2147483647, -1, 2816, 832, 1472},
            {-1, 2147483647, 3840, 5632, 832, 1472}, {-1, 2147483647, 7168, 2147483647, 832, 1472},
            {-1, 1536, -1, 2048, 1472, 2147483647}, {-1, 3504, 2048, 2147483647, 1472, 2147483647},
            {3840, 2147483647, -1, 21888, 1472, 6400}, {3840, 2147483647, -1, 2147483647, 6400, 2147483647}}},
    {4.0,
        {{3840, 2147483647, 21888, 2147483647, 1472, 6400}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BTwoRankFP16PvalueMap = {
    {1,
        {{-1, 9216, -1, 2147483647, -1, 1152}, {-1, 9216, 4480, 2147483647, 1152, 1488},
            {-1, 9216, -1, 2147483647, 1488, 1920}, {9216, 2147483647, 2112, 7168, -1, 192},
            {9216, 24576, 7168, 2147483647, -1, 1920}, {-1, 9984, -1, 9216, 1920, 3760},
            {-1, 9984, 9216, 2147483647, 1920, 11264}, {9984, 2147483647, 14080, 2147483647, 1920, 11264}}},
    {4,
        {{-1, 9216, -1, 4480, 1152, 1488}, {9216, 81920, -1, 2112, -1, 192},
            {9216, 2147483647, -1, 7168, 192, 1920}, {24576, 2147483647, 7168, 2147483647, -1, 1920},
            {-1, 9984, -1, 9216, 3760, 9216}, {9984, 2147483647, -1, 14080, 1920, 11264},
            {-1, 2147483647, -1, 2147483647, 11264, 2147483647}}},
    {5,
        {{81920, 2147483647, -1, 2112, -1, 192}}},
    {8,
        {{-1, 9984, -1, 9216, 9216, 11264}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BTwoRankFP16CommnpusplitMap = {
    {2,
     {{-1, 2147483647, -1, 2147483647, -1, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BTwoRankFP16M0Map = {
    {256,
        {{-1, 3328, -1, 2147483647, -1, 832}, {3328, 2147483647, -1, 2147483647, -1, 192},
            {3328, 2147483647, -1, 2147483647, 576, 832}, {-1, 2147483647, 38912, 2147483647, 832, 2147483647}}},
    {128,
        {{3328, 2147483647, -1, 2147483647, 192, 576}, {-1, 2147483647, -1, 38912, 832, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BFourRankFP16UbmovenumMap = {
    {8.0,
        {{-1, 7552, -1, 2147483647, -1, 72}}},
    {6.0,
        {{-1, 7552, -1, 2147483647, 72, 448}}},
    {4.0,
        {{-1, 7552, -1, 13312, 448, 14016}, {-1, 7552, -1, 2147483647, 14016, 2147483647},
            {7552, 2147483647, -1, 6656, 72, 3584}, {7552, 81920, 6656, 9088, 72, 3584},
            {7552, 2147483647, -1, 9088, 3584, 2147483647}}},
    {2.0,
        {{-1, 7552, 13312, 2147483647, 448, 14016}, {81920, 2147483647, 6656, 9088, 72, 3584},
            {7552, 2147483647, 9088, 2147483647, 72, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BFourRankFP16PvalueMap = {
    {1,
        {{-1, 9216, -1, 2147483647, -1, 1152}, {-1, 9216, 4480, 2147483647, 1152, 2352},
            {-1, 9216, -1, 2147483647, 2352, 4000}, {9216, 24576, 850, 2147483647, -1, 1152},
            {24576, 81920, 6656, 2147483647, -1, 4000}, {-1, 12332, 7552, 19840, 4000, 4288},
            {-1, 2147483647, 19840, 2147483647, 4000, 4288}, {-1, 17152, 8000, 9088, 4544, 2147483647},
            {-1, 17152, 11136, 2147483647, 4544, 2147483647}, {21248, 28672, -1, 2147483647, 4544, 9472}}},
    {4,
        {{-1, 9216, -1, 4480, 1152, 2352}, {9216, 2147483647, -1, 64, -1, 72},
            {273408, 2147483647, -1, 850, 72, 4000}, {9216, 24576, 850, 2147483647, 1152, 4000},
            {24576, 81920, 850, 6656, -1, 4000}, {81920, 2147483647, 850, 2147483647, -1, 4000},
            {-1, 2147483647, -1, 7552, 4000, 4288}, {12332, 2147483647, 7552, 19840, 4000, 4288},
            {-1, 2147483647, -1, 2147483647, 4288, 4544}, {-1, 10240, -1, 1152, 4544, 2147483647},
            {-1, 17152, 1152, 8000, 4544, 2147483647}, {-1, 17152, 9088, 11136, 4544, 2147483647}}},
    {6,
        {{9216, 2147483647, 64, 850, -1, 72}, {17152, 28672, -1, 2147483647, 9472, 14336}}},
    {5,
        {{9216, 273408, -1, 850, 72, 4000}, {10240, 17152, -1, 1152, 4544, 2147483647},
            {17152, 21248, -1, 2147483647, 4544, 9472}, {17152, 28672, -1, 2147483647, 14336, 2147483647},
            {28672, 2147483647, -1, 2147483647, 4544, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BFourRankFP16CommdatasplitMap = {
    {4,
        {{-1, 19456, -1, 2147483647, -1, 448}, {19456, 2147483647, -1, 2147483647, -1, 72}}},
    {20,
        {{19456, 2147483647, -1, 2147483647, 72, 448}, {-1, 2147483647, -1, 2147483647, 448, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BFourRankFP16CommnpusplitMap = {
    {4,
        {{-1, 19456, -1, 2147483647, -1, 448}, {19456, 2147483647, -1, 2147483647, -1, 72}}},
    {1,
        {{19456, 2147483647, -1, 2147483647, 72, 448}, {-1, 2147483647, -1, 2147483647, 448, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BFourRankFP16M0Map = {
    {256,
        {{-1, 2147483647, -1, 2147483647, -1, 192}, {-1, 4608, -1, 2147483647, 448, 832},
            {-1, 4608, 38912, 2147483647, 832, 2147483647}}},
    {128,
        {{-1, 4608, -1, 2147483647, 192, 448}, {-1, 4608, -1, 38912, 832, 2147483647},
            {4608, 2147483647, -1, 2147483647, 192, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BEightRankFP16UbmovenumMap = {
    {8.0,
        {{-1, 13312, -1, 2147483647, -1, 192}, {-1, 2147483647, -1, 2147483647, 192, 2147483647}}},
    {32.0,
        {{13312, 2147483647, -1, 2147483647, -1, 192}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BEightRankFP16PvalueMap = {
    {14,
        {{-1, 13312, -1, 2147483647, -1, 192}}},
    {4,
        {{13312, 22528, -1, 2147483647, -1, 72}, {-1, 3072, -1, 3968, 192, 9216},
            {3072, 15360, -1, 3968, 192, 4608}, {-1, 15360, 3968, 4608, 4608, 2147483647},
            {15360, 2147483647, 2736, 7552, 192, 2147483647}}},
    {8,
        {{22528, 2147483647, -1, 2147483647, -1, 72}, {13312, 2147483647, -1, 2147483647, 72, 192},
            {15360, 2147483647, 1878, 2736, 192, 2147483647}}},
    {6,
        {{-1, 3072, -1, 3968, 9216, 2147483647}, {3072, 15360, -1, 3968, 4608, 2147483647},
            {15360, 2147483647, -1, 1878, 192, 2147483647}}},
    {1,
        {{-1, 15360, 3968, 4608, 192, 4608}, {-1, 15360, 4608, 2147483647, 192, 2147483647},
            {15360, 2147483647, 7552, 2147483647, 192, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BEightRankFP16CommdatasplitMap = {
    {20,
        {{-1, 13312, -1, 2147483647, -1, 72}}},
    {16,
        {{13312, 2147483647, -1, 2147483647, -1, 72}, {-1, 2147483647, -1, 2147483647, 72, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BEightRankFP16CommnpusplitMap = {
    {1,
     {{-1, 2147483647, -1, 2147483647, -1, 2147483647}}}
};

static std::map<int, std::vector<std::vector<int>>> g_allreduce910BEightRankFP16M0Map = {
    {256,
        {{-1, 2147483647, -1, 2147483647, -1, 192}, {-1, 4608, -1, 2147483647, 448, 832},
            {-1, 4608, 38912, 2147483647, 832, 2147483647}}},
    {128,
        {{-1, 4608, -1, 2147483647, 192, 448}, {-1, 4608, -1, 38912, 832, 2147483647},
            {4608, 2147483647, -1, 2147483647, 192, 2147483647}}}
};

const LUTGroup AllReduce2p{
    ALLREDUCE_910B_TWO_RANK_FP16_M0_DEFAULT,
    ALLREDUCE_910B_TWO_RANK_FP16_PVALUE_DEFAULT,
    ALLREDUCE_910B_TWO_RANK_FP16_UBMOVENUM_DEFAULT,
    ALLREDUCE_910B_TWO_RANK_FP16_COMMNPUSPLIT_DEFAULT,
    ALLREDUCE_910B_TWO_RANK_FP16_COMMDATASPLIT_DEFAULT,
    g_allreduce910BTwoRankFP16M0Map,
    g_allreduce910BTwoRankFP16PvalueMap,
    g_allreduce910BTwoRankFP16UbmovenumMap,
    g_allreduce910BTwoRankFP16CommnpusplitMap,
    g_allreduce910BTwoRankFP16CommdatasplitMap
};

const LUTGroup AllReduce4p{
    ALLREDUCE_910B_FOUR_RANK_FP16_M0_DEFAULT,
    ALLREDUCE_910B_FOUR_RANK_FP16_PVALUE_DEFAULT,
    ALLREDUCE_910B_FOUR_RANK_FP16_UBMOVENUM_DEFAULT,
    ALLREDUCE_910B_FOUR_RANK_FP16_COMMNPUSPLIT_DEFAULT,
    ALLREDUCE_910B_FOUR_RANK_FP16_COMMDATASPLIT_DEFAULT,
    g_allreduce910BFourRankFP16M0Map,
    g_allreduce910BFourRankFP16PvalueMap,
    g_allreduce910BFourRankFP16UbmovenumMap,
    g_allreduce910BFourRankFP16CommnpusplitMap,
    g_allreduce910BFourRankFP16CommdatasplitMap
};

const LUTGroup AllReduce8p{
    ALLREDUCE_910B_EIGHT_RANK_FP16_M0_DEFAULT,
    ALLREDUCE_910B_EIGHT_RANK_FP16_PVALUE_DEFAULT,
    ALLREDUCE_910B_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT,
    ALLREDUCE_910B_EIGHT_RANK_FP16_COMMNPUSPLIT_DEFAULT,
    ALLREDUCE_910B_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT,
    g_allreduce910BEightRankFP16M0Map,
    g_allreduce910BEightRankFP16PvalueMap,
    g_allreduce910BEightRankFP16UbmovenumMap,
    g_allreduce910BEightRankFP16CommnpusplitMap,
    g_allreduce910BEightRankFP16CommdatasplitMap
};