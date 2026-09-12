/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCOC_H
#define CATCOC_H

#include <kernel_operator.h>

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"

namespace Catcoc {

template <typename Index, typename LongIndex, int RANK>
CATLASS_HOST_DEVICE
LongIndex Numel(Catlass::Coord<RANK, Index, LongIndex> const &coord)
{
    LongIndex product = 1;
    for (int i = 0; i < RANK; ++i) {
        product *= static_cast<LongIndex>(coord[i]);
    }
    return product;
}

template <typename Index, typename LongIndex, int RANK>
CATLASS_HOST_DEVICE
LongIndex Dot(Catlass::Coord<RANK, Index> const &coord, Catlass::Coord<RANK, LongIndex> const &stride,
    LongIndex accumulator = {})
{
    for (int i = 0; i < RANK; ++i) {
        accumulator += static_cast<LongIndex>(coord[i]) * stride[i];
    }
    return accumulator;
}

template <class T>
CATLASS_HOST_DEVICE constexpr
T Min(T const &lhs, T const &rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

template <typename Index, int RANK>
CATLASS_HOST_DEVICE constexpr
auto Min(Catlass::Coord<RANK, Index> const &lhs, Catlass::Coord<RANK, Index> const &rhs)
{
    Catlass::Coord<RANK, Index> result;
    for (int i = 0; i < RANK; ++i) {
        result[i] = Min(lhs[i], rhs[i]);
    }
    return result;
}

template <class T>
CATLASS_HOST_DEVICE constexpr
T Max(T const &lhs, T const &rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

template <typename Index, int RANK>
CATLASS_HOST_DEVICE constexpr
auto Max(Catlass::Coord<RANK, Index> const &lhs, Catlass::Coord<RANK, Index> const &rhs)
{
    Catlass::Coord<RANK, Index> result;
    for (int i = 0; i < RANK; ++i) {
        result[i] = Max(lhs[i], rhs[i]);
    }
    return result;
}

namespace layout {

template <int RANK_, typename Index_=int32_t>
struct AffineRankN {
public:
    static int const RANK = RANK_;
    using Index = Index_;
    using LongIndex = int64_t;
    using TensorCoord = Catlass::Coord<RANK, Index>;
    using Stride = Catlass::Coord<RANK, LongIndex>;

    CATLASS_HOST_DEVICE
    AffineRankN(Stride const &stride = Stride()) : stride_(stride) {}

    CATLASS_HOST_DEVICE
    static AffineRankN Packed(TensorCoord const &extent)
    {
        AffineRankN layout;
        layout.stride_[RANK - 1] = 1;

        for (int i = RANK - 1; i > 0; --i) {
            layout.stride_[i - 1] = layout.stride_[i] * extent[i];
        }

        return layout;
    }

    CATLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const &coord) const
    {
        return Dot(coord, stride_);
    }

private:
    Stride stride_;
};

}

namespace Padding {
static constexpr uint32_t STRIDE_ALIGNMENT_THRESHOLD = 65536;

template <class Layout> size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
{
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
           RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

CATLASS_HOST_DEVICE
bool IsNeedPadding(Catlass::layout::RowMajor layout, uint32_t align)
{
    if (align == 0) {
        return false;
    }
    if (layout.stride(0) < STRIDE_ALIGNMENT_THRESHOLD) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

CATLASS_HOST_DEVICE
bool IsNeedPadding(Catlass::layout::ColumnMajor layout, uint32_t align)
{
    if (align == 0) {
        return false;
    }
    if (layout.stride(1) < STRIDE_ALIGNMENT_THRESHOLD) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}

template <class Type, bool PADDING> struct PaddingHelper {
    using ArchTag = typename Catlass::Arch::AtlasA2;
    using Layout = typename Type::Layout;
    using Element = typename Type::Element;

    using LayoutPadding = std::conditional_t<std::is_same_v<Layout, Catlass::layout::RowMajor>,
                                             Catlass::layout::PaddingRowMajor,
                                             Catlass::layout::PaddingColumnMajor>;
    using ActualType = std::conditional_t<PADDING, Catlass::Gemm::GemmType<Element, LayoutPadding>, Type>;
    static const uint32_t COMPUTE_LENGTH = 96 * 1024 / sizeof(Element);
    using GlobalPadding = std::conditional_t<
        PADDING, Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag,
        Element, Layout, LayoutPadding, COMPUTE_LENGTH>, void>;
    using LayoutW = std::conditional_t<PADDING, LayoutPadding, Layout>;

    CATLASS_DEVICE
    static LayoutW GetLayoutW(uint32_t a, uint32_t b, uint32_t padA, uint32_t padB)
    {
        if constexpr (PADDING) {
            LayoutPadding layoutW = LayoutPadding(a, b, padA, padB);
            return layoutW;
        } else {
            Layout layoutW = Layout(a, b);
            return layoutW;
        }
    }
};

}

} // namespace Catcoc

#endif // CATCOC_H