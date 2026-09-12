/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCOC_DIST_COORD_H
#define CATCOC_DIST_COORD_H

#pragma once

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/coord.hpp"

namespace Catcoc {
static constexpr uint32_t two_dim_coord_dimensions = 2;
static constexpr uint32_t matrix_coord_dimensions = 3;
static constexpr uint32_t gemm_coord_dimensions = 4;

struct DistMatrixCoord : public Catlass::Coord<matrix_coord_dimensions, uint32_t> {
    using Index = uint32_t;
    using Base = Catlass::Coord<matrix_coord_dimensions, Index>;
    static constexpr uint32_t ROW_INDEX = 0;
    static constexpr uint32_t COLUMN_INDEX = 1;
    static constexpr int RANK_INDEX = 2;

    CATLASS_HOST_DEVICE
    DistMatrixCoord() = default;

    CATLASS_HOST_DEVICE
    DistMatrixCoord(Base const &coord) : Base(coord)
    {
    }

    CATLASS_HOST_DEVICE
    DistMatrixCoord(Index row, Index column, Index rank) : Base(Catlass::MakeCoord<Index>(row, column, rank))
    {
    }

    CATLASS_HOST_DEVICE
    DistMatrixCoord(Catlass::Coord<two_dim_coord_dimensions, Index> matrixCoord, Index rank)
        : Base(Catlass::MakeCoord<Index>(matrixCoord[0], matrixCoord[1], rank))
    {
    }

    CATLASS_HOST_DEVICE
    Index const &row() const
    {
        return this->At(ROW_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &row()
    {
        return this->At(ROW_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &column() const
    {
        return this->At(COLUMN_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &column()
    {
        return this->At(COLUMN_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &rank() const
    {
        return this->At(RANK_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &rank()
    {
        return this->At(RANK_INDEX);
    }

    CATLASS_HOST_DEVICE
    auto GetCoordInRank() const
    {
        return this->GetCoordByAxis<ROW_INDEX, COLUMN_INDEX>();
    }
};

struct DistGemmCoord : public Catlass::Coord<gemm_coord_dimensions, uint32_t> {
    using Index = uint32_t;
    using Base = Catlass::Coord<gemm_coord_dimensions, Index>;
    static constexpr int M_INDEX = 0;
    static constexpr int N_INDEX = 1;
    static constexpr int K_INDEX = 2;
    static constexpr int RANK_INDEX = 3;

    CATLASS_HOST_DEVICE
    DistGemmCoord() = default;

    CATLASS_HOST_DEVICE
    DistGemmCoord(Base const &coord) : Base(coord)
    {
    }

    CATLASS_HOST_DEVICE
    DistGemmCoord(Index m, Index n, Index k, Index rank) : Base(Catlass::MakeCoord<Index>(m, n, k, rank))
    {
    }

    CATLASS_HOST_DEVICE
    Index const &m() const
    {
        return this->At(M_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &n() const
    {
        return this->At(N_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &k() const
    {
        return this->At(K_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &rank() const
    {
        return this->At(RANK_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &m()
    {
        return this->At(M_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &n()
    {
        return this->At(N_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &k()
    {
        return this->At(K_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &rank()
    {
        return this->At(RANK_INDEX);
    }

    CATLASS_HOST_DEVICE
    auto GetCoordMN() const
    {
        return this->GetCoordByAxis<M_INDEX, N_INDEX>();
    }

    CATLASS_HOST_DEVICE
    auto GetCoordMK() const
    {
        return this->GetCoordByAxis<M_INDEX, K_INDEX>();
    }

    CATLASS_HOST_DEVICE
    auto GetCoordKN() const
    {
        return this->GetCoordByAxis<K_INDEX, N_INDEX>();
    }

    CATLASS_HOST_DEVICE
    auto GetCoordMNK() const
    {
        return this->GetCoordByAxis<M_INDEX, N_INDEX, K_INDEX>();
    }
};

}  // namespace Catcoc

#endif  // CATCOC_DIST_COORD_H