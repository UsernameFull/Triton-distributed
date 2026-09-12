/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCOC_EPILOGUE_TILE_TILE_REMOTE_COPY_H
#define CATCOC_EPILOGUE_TILE_TILE_REMOTE_COPY_H

#include "catcoc/catcoc.h"
#include "catcoc/detail/remote_copy_type.h"

// from catlass
#include "catlass/catlass.hpp"

// from shmem
#include "shmem.h"

namespace Catcoc::CommEpilogue::Tile {

using Catlass::MatrixCoord;

template <
    /// Tag indicating architecture
    class ArchTag,
    class SrcType_,
    class DstType_,
    detail::CopyDirect CopyDirect_
>
struct TileRemoteCopy {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported tile copy, can not find the specialization.");
};

template <
    class ArchTag,
    class SrcType_,
    class DstType_
>
struct TileRemoteCopy<ArchTag, SrcType_, DstType_, detail::CopyDirect::Get> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Get;

    CATLASS_DEVICE
    TileRemoteCopy() {}

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementDst> const &dstTensor, LayoutDst const &dstLayout,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor, LayoutDst const &srcLayout,
        MatrixCoord const &copyShape,
        AscendC::LocalTensor<ElementSrc> const &tmpUb,
        uint32_t copyEventId,
        uint32_t peerIdx
    )
    {
        non_contiguous_copy_param copyParams;
        copyParams.repeat = copyShape.row();
        copyParams.length = copyShape.column();
        copyParams.src_ld = srcLayout.stride(0);
        copyParams.dst_ld = dstLayout.stride(0);
        aclshmemx_mte_get_nbi(dstTensor, srcTensor, tmpUb, copyParams, peerIdx, copyEventId);
    }
};

template <
    class ArchTag,
    class SrcType_,
    class DstType_
>
struct TileRemoteCopy<ArchTag, SrcType_, DstType_, detail::CopyDirect::Put> {
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    static constexpr detail::CopyDirect RemoteCopyDirect = detail::CopyDirect::Put;

    CATLASS_DEVICE
    TileRemoteCopy() {}

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementDst> const &dstTensor, LayoutDst const &dstLayout,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor, LayoutDst const &srcLayout,
        MatrixCoord const &copyShape,
        AscendC::LocalTensor<ElementSrc> const &tmpUb,
        uint32_t copyEventId,
        uint32_t peerIdx
    )
    {
        non_contiguous_copy_param copyParams;
        copyParams.repeat = copyShape.row();
        copyParams.length = copyShape.column();
        copyParams.src_ld = srcLayout.stride(0);
        copyParams.dst_ld = dstLayout.stride(0);
        aclshmemx_mte_put_nbi(dstTensor, srcTensor, tmpUb, copyParams, peerIdx, copyEventId);
    }
};

} // namespace Catlass::Epilogue::Tile

#endif // CATCOC_EPILOGUE_TILE_TILE_REMOTE_COPY_H
