//===- Utils.cpp - Utilities to support the Tensor dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Tensor dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::tensor;

PadOp mlir::tensor::createPadHighOp(RankedTensorType type, Value source,
                                    Value pad, bool nofold, Location loc,
                                    OpBuilder &b) {
  SmallVector<OpFoldResult> low(type.getRank(), b.getIndexAttr(0));
  SmallVector<OpFoldResult> high(type.getRank(), b.getIndexAttr(0));
  for (const auto &en : enumerate(type.getShape())) {
    // Pad only the static dimensions of the result tensor type.
    if (ShapedType::isDynamic(en.value()))
      continue;
    // Compute the padding width.
    AffineExpr d0;
    bindDims(b.getContext(), d0);
    OpFoldResult sz = tensor::getMixedSize(b, loc, source, en.index());
    high[en.index()] =
        affine::makeComposedFoldedAffineApply(b, loc, en.value() - d0, {sz});
  }
  return b.create<PadOp>(loc, type, source, low, high, pad, nofold);
}

SmallVector<Value> mlir::tensor::createDynamicDimValues(OpBuilder &b,
                                                        Location loc,
                                                        Value rankedTensor) {
  auto tensorTy = cast<RankedTensorType>(rankedTensor.getType());
  SmallVector<Value> dynamicDims;
  for (const auto &en : llvm::enumerate(tensorTy.getShape())) {
    if (en.value() == ShapedType::kDynamic)
      dynamicDims.push_back(
          b.create<tensor::DimOp>(loc, rankedTensor, en.index()));
  }
  return dynamicDims;
}

FailureOr<RankedTensorType>
mlir::tensor::computeTransposedType(RankedTensorType rankedTensorType,
                                    ArrayRef<int64_t> transposeVector) {
  if (transposeVector.empty())
    return rankedTensorType;

  if (!isPermutationVector(transposeVector) ||
      transposeVector.size() != static_cast<size_t>(rankedTensorType.getRank()))
    return failure();

  SmallVector<int64_t> transposedShape(rankedTensorType.getShape().begin(),
                                       rankedTensorType.getShape().end());
  applyPermutationToVector(transposedShape, transposeVector);

  using RTTBuilder = RankedTensorType::Builder;
  RankedTensorType transposedTensorType =
      RTTBuilder(rankedTensorType).setShape(transposedShape);
  return transposedTensorType;
}

/// The permutation can be obtained from two permutations:
///   a) Compute the permutation vector to move the last `numPackedDims` into
///      the `innerPosDims` of a shape of rank `rank`.
///   b) Compute the permutation vector to move outer dims if the
///      `outerPerm` parameter is not empty.
/// Apply (b) permutation on (a) permutation to get the final permutation.
static SmallVector<int64_t>
computePackUnPackPerm(int64_t rank, ArrayRef<int64_t> &innerDimsPos,
                      ArrayRef<int64_t> &outerPerm,
                      PackingMetadata &packingMetadata) {
  int64_t numPackedDims = innerDimsPos.size();
  auto lastDims =
      llvm::to_vector(llvm::seq<int64_t>(rank - numPackedDims, rank));
  packingMetadata = computePackingMetadata(rank, innerDimsPos);
  SmallVector<int64_t> innerPositionsPerm =
      computePermutationVector(rank, lastDims, packingMetadata.insertPositions);

  SmallVector<int64_t> outerPos = packingMetadata.outerPositions;
  if (!outerPerm.empty())
    applyPermutationToVector(outerPos, outerPerm);
  SmallVector<int64_t> outerPositionPerm =
      computePermutationVector(rank, packingMetadata.outerPositions, outerPos);

  SmallVector<int64_t> packInverseDestPermutation = innerPositionsPerm;
  applyPermutationToVector(packInverseDestPermutation, outerPositionPerm);
  return packInverseDestPermutation;
}

SmallVector<int64_t> mlir::tensor::getPackInverseDestPerm(PackOp packOp) {

  PackingMetadata pMetadata;
  int64_t packedRank = packOp.getDestType().getRank();
  ArrayRef<int64_t> innerDimPos = packOp.getInnerDimsPos();
  ArrayRef<int64_t> outerPerm = packOp.getOuterDimsPerm();
  SmallVector<int64_t> packInvDestPerm =
      computePackUnPackPerm(packedRank, innerDimPos, outerPerm, pMetadata);
  return packInvDestPerm;
}

SmallVector<int64_t> mlir::tensor::getUnPackInverseSrcPerm(UnPackOp unpackOp) {
  PackingMetadata metadata;
  return mlir::tensor::getUnPackInverseSrcPerm(unpackOp, metadata);
}

SmallVector<int64_t>
mlir::tensor::getUnPackInverseSrcPerm(UnPackOp unpackOp,
                                      PackingMetadata &metadata) {
  int64_t unpackRank = unpackOp.getSourceType().getRank();
  ArrayRef<int64_t> innerDimPos = unpackOp.getInnerDimsPos();
  ArrayRef<int64_t> outerPerm = unpackOp.getOuterDimsPerm();
  SmallVector<int64_t> unpackInvSrcPerm =
      computePackUnPackPerm(unpackRank, innerDimPos, outerPerm, metadata);
  return unpackInvSrcPerm;
}

bool mlir::tensor::isCastLikeInsertSliceOp(InsertSliceOp op) {
  llvm::SmallBitVector droppedDims = op.getDroppedDims();
  int64_t srcDim = 0;
  RankedTensorType resultType = op.getDestType();
  // Source dims and destination dims (apart from dropped dims) must have the
  // same size.
  for (int64_t resultDim = 0; resultDim < resultType.getRank(); ++resultDim) {
    if (droppedDims.test(resultDim)) {
      // InsertSlice may expand unit dimensions that result from inserting a
      // size-1 slice into a non-size-1 result dimension.
      if (resultType.getDimSize(resultDim) != 1)
        return false;
      continue;
    }
    FailureOr<bool> equalDimSize = ValueBoundsConstraintSet::areEqual(
        {op.getSource(), srcDim}, {op.getResult(), resultDim});
    if (failed(equalDimSize) || !*equalDimSize)
      return false;
    ++srcDim;
  }

  return true;
}

bool mlir::tensor::isCastLikeExtractSliceOp(ExtractSliceOp op) {
  llvm::SmallBitVector droppedDims = op.getDroppedDims();
  int64_t resultDim = 0;
  // Source dims and result dims (apart from dropped dims) must have the same
  // size.
  RankedTensorType sourceType = op.getSourceType();
  for (int64_t dim = 0, e = sourceType.getRank(); dim < e; ++dim) {
    if (droppedDims.test(dim)) {
      // ExtractSlice may drop unit dimensions that result from taking a size-1
      // slice from a non-size-1 source dimension.
      if (sourceType.getDimSize(dim) != 1)
        return false;
      continue;
    }
    FailureOr<bool> equalDimSize = ValueBoundsConstraintSet::areEqual(
        {op.getSource(), dim}, {op.getResult(), resultDim});
    if (failed(equalDimSize) || !*equalDimSize)
      return false;
    ++resultDim;
  }

  return true;
}

#if BSPUB_DAVINCI_BISHENGIR
static bool hasNonOneStride(const SmallVector<OpFoldResult> &strides) {
  return llvm::any_of(strides, [](OpFoldResult ofr) {
    auto intMaybe = getConstantIntValue(ofr);
    if (!intMaybe.has_value()) {
      return true;
    }
    return intMaybe.value() != 1;
  });
}

bool mlir::tensor::isOffsetBytesAligned(tensor::ExtractSliceOp sliceOp,
                                        int64_t bytesToAlign) {
  TensorType srcType = cast<TensorType>(sliceOp.getSource().getType());
  unsigned elemSizeInBytes = srcType.getElementTypeBitWidth() / 8;

  SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> strides = sliceOp.getMixedStrides();
  if (offsets.empty()) {
    // no offset means zero offset, i.e. aligned
    return true;
  }
  if (hasNonOneStride(strides)) {
    // strides should only be one
    return false;
  }
  ShapedType shapeType = llvm::cast<ShapedType>(sliceOp.getSource().getType());
  ArrayRef<int64_t> shapes = shapeType.getShape();

  // init accumOffset to 0
  // init accumShape to 1
  // accumOffset[i] = accumOffset[i+1] + offsets[i] * accumShape[i+1]
  int64_t accumShape = 1;
  int64_t accumOffset = 0;
  for (int idx = offsets.size() - 1; idx >= 0; idx--) {
    auto offsetMaybe = getConstantIntValue(offsets[idx]);
    if (!offsetMaybe.has_value()) {
      // offset must be constant to check alignment
      return false;
    }
    int64_t curDimOffset = offsetMaybe.value();
    accumOffset = accumOffset + curDimOffset * accumShape;
    if ((accumOffset * elemSizeInBytes) % bytesToAlign != 0) {
      // accum offset must be bytes aligned
      return false;
    }
    int64_t curShape = shapes[idx];
    if (ShapedType::isDynamic(curShape)) {
      // shape size must be constant to check alignment
      return false;
    }
    accumShape = accumShape * curShape;
  }
  return true;
}
#endif
