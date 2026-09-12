//===- ExpandShapeUtils.h - Helpers related to expand shape ops -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities and common canonicalization patterns for
// expand shape operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {

template <typename ExpandShapeOpTy, typename CastOpTy>
struct FoldConstantDimOfOutputShape : public OpRewritePattern<ExpandShapeOpTy> {
  using OpRewritePattern<ExpandShapeOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpandShapeOpTy expandOp,
                                PatternRewriter &rewriter) const final {
    // no constant to fold
    ValueRange dynamicShapes = expandOp.getOutputShape();
    if (dynamicShapes.empty()) {
      return failure();
    }

    // fold constant dynamic output shape values into static shape values
    ArrayRef<int64_t> staticShapes = expandOp.getStaticOutputShape();
    SmallVector<OpFoldResult> ofrs =
        getMixedValues(staticShapes, dynamicShapes, rewriter);
    LogicalResult fold = foldDynamicIndexList(ofrs);
    if (failed(fold)) {
      return failure();
    }
    SmallVector<Value> foldedDynamicShape;
    SmallVector<int64_t> foldedStaticShapes;
    dispatchIndexOpFoldResults(ofrs, foldedDynamicShape, foldedStaticShapes);

    // clarify the reassociation type for overload resolution
    SmallVector<ReassociationIndices> reassociation =
        expandOp.getReassociationIndices();
    auto resultType = expandOp.getResultType();
    auto foldedResultType = resultType.clone(foldedStaticShapes);
    auto srcType = expandOp.getSrcType();

    // infer the src shape based on the folded result shape
    ShapedType foldedSrcType;
    if constexpr (std::is_same_v<ExpandShapeOpTy, tensor::ExpandShapeOp>) {
      RankedTensorType tensorTy = cast<RankedTensorType>(foldedResultType);
      foldedSrcType =
          tensor::CollapseShapeOp::inferCollapsedType(tensorTy, reassociation);
    } else if constexpr (std::is_same_v<ExpandShapeOpTy,
                                        memref::ExpandShapeOp>) {
      MemRefType memrefTy = cast<MemRefType>(foldedResultType);
      foldedSrcType = memref::CollapseShapeOp::computeCollapsedType(
          memrefTy, reassociation);
    } else {
      // only support ExpandShapeOp for tensor and memref
      return failure();
    }

    // if src shape is changed, insert cast between src and op for compatibility
    Value castFromSource = expandOp.getSrc();
    if (srcType != foldedSrcType) {
      castFromSource = rewriter.create<CastOpTy>(
          expandOp.getSrc().getLoc(), foldedSrcType, expandOp.getSrc());
    }

    // create new op with folded result shapes and (maybe new) source
    auto newOp =
        rewriter.create<ExpandShapeOpTy>(expandOp.getLoc(), foldedResultType,
                                         castFromSource, reassociation, ofrs);
    // if resultType of new op is changed, insert cast between new op and all
    // uses of origin op for compatibility
    Value castToResult = newOp.getResult();
    if (castToResult.getType() != expandOp.getType()) {
      castToResult = rewriter.create<CastOpTy>(expandOp.getLoc(),
                                               expandOp.getType(), newOp);
    }
    rewriter.replaceOp(expandOp, castToResult);
    return success();
  }
};

} // namespace mlir
