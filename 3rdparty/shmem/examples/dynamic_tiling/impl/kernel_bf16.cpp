/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "impl/kernel/matmul_allreduce.h"
#include "impl/kernel/allgather_matmul.h"
#include "impl/kernel/matmul_reduce_scatter.h"
#include "impl/kernel/allgather_matmul_with_gather_result.h"
#include "impl/kernel/matmul_reduce_scatter_padding_ab.h"
#include "impl/kernel/matmul_reduce_scatter_padding_a.h"
#include "impl/kernel/matmul_reduce_scatter_padding_b.h"
#include "impl/kernel/allgather_matmul_padding.h"

using namespace AscendC;

using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementC = bfloat16_t;

using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::RowMajor;

using LayoutA1 = Catlass::layout::ColumnMajor;
using LayoutB1 = Catlass::layout::ColumnMajor;

using LayoutC = Catlass::layout::RowMajor;

void LaunchMatmulAllReduceBF16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *gatherA, uint8_t *workspace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)gatherA;
    (void)workspace;
    if (!transA && !transB) {
        MatmulAllReduce<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        MatmulAllReduce<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else if (transA && !transB) {
        MatmulAllReduce<ElementA, LayoutA1, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else {
        MatmulAllReduce<ElementA, LayoutA1, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    }
}

void LaunchAllGatherMatmulBF16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *gatherA, uint8_t *workspace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)gatherA;
    (void)workspace;
    if (!transA && !transB) {
        AllGatherMatmul<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        AllGatherMatmul<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    }
}

void LaunchMatmulReduceScatterBF16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *gatherA, uint8_t *workspace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)gatherA;
    (void)workspace;
    if (!transA && !transB) {
        MatmulReduceScatter<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        MatmulReduceScatter<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else if (transA && !transB) {
        MatmulReduceScatter<ElementA, LayoutA1, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else {
        MatmulReduceScatter<ElementA, LayoutA1, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    }
}

void LaunchAllGatherMatmulWithGatherResultBF16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *gatherA, uint8_t *workspace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workspace;
    if (!transA && !transB) {
        AllGatherMatmulWithGatherResult<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, gatherA, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        AllGatherMatmulWithGatherResult<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, gatherA, symmetricPtr, cocTiling);
    }
}

void LaunchMatmulReduceScatterPaddingABBF16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *aWorkSpace, uint8_t *bWorkSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    if (!transA && !transB) {
        MatmulReduceScatterPaddingAB<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        MatmulReduceScatterPaddingAB<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    } else if (transA && !transB) {
        MatmulReduceScatterPaddingAB<ElementA, LayoutA1, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    } else {
        MatmulReduceScatterPaddingAB<ElementA, LayoutA1, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    }
}

void LaunchMatmulReduceScatterPaddingABF16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *aWorkSpace, uint8_t *bWorkSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    if (!transA && !transB) {
        MatmulReduceScatterPaddingA<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        MatmulReduceScatterPaddingA<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    } else if (transA && !transB) {
        MatmulReduceScatterPaddingA<ElementA, LayoutA1, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    } else {
        MatmulReduceScatterPaddingA<ElementA, LayoutA1, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    }
}

void LaunchMatmulReduceScatterPaddingBBF16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *aWorkSpace, uint8_t *bWorkSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    if (!transA && !transB) {
        MatmulReduceScatterPaddingB<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        MatmulReduceScatterPaddingB<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    } else if (transA && !transB) {
        MatmulReduceScatterPaddingB<ElementA, LayoutA1, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    } else {
        MatmulReduceScatterPaddingB<ElementA, LayoutA1, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, aWorkSpace, bWorkSpace, symmetricPtr, cocTiling);
    }
}

void LaunchAllGatherMatmulPaddingBF16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *gatherA, uint8_t *workspace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)gatherA;
    if (!transA && !transB) {
        AllGatherMatmulPadding<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, workspace, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        AllGatherMatmulPadding<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, workspace, symmetricPtr, cocTiling);
    }
}