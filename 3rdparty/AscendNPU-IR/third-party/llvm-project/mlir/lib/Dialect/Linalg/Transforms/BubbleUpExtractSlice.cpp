//===- BubbleUpExtractSlice.cpp - bubble up tensor.extract_slice ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns that transforms linalg.<op> +
// tensor.extract_slice into tensor.extract_slice + linalg.<op> to reduce
// the computation for the linalg op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Config/mlir-config.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#if BSPUB_DAVINCI_BISHENGIR
#define DEBUG_TYPE "bubble-up-extract-slice"
#endif

using namespace mlir;
using namespace mlir::linalg;

namespace {
/// Bubble up extract_slice above Linalg operation.
///
/// A sequence of operations
///
/// ```mlir
/// %0 = linalg.<op> ... arg0, arg1, ...
/// %1 = tensor.extract_slice %0 ...
/// ```
///
/// can be replaced with
///
/// ```mlir
/// %0 = tensor.extract_slice %arg0
/// %1 = tensor.extract_slice %arg1
/// %2 = linalg.<op> ... %0, %1, ...
/// ```
///
/// This results in the reduce computation of the linalg operation.
///
struct BubbleUpExtractSliceOpPattern
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

#if BSPUB_DAVINCI_BISHENGIR
public:
  BubbleUpExtractSliceOpPattern(MLIRContext *ctx,
                                const BubbleUpExtractSliceOptions &options)
      : OpRewritePattern<tensor::ExtractSliceOp>(ctx), options(options) {}

  LogicalResult bubbleUpBroadcast(tensor::ExtractSliceOp sliceOp,
                                  PatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "Debug broadcast " << sliceOp << "\n";);
    auto broadcastOp =
        dyn_cast<linalg::BroadcastOp>(sliceOp.getSource().getDefiningOp());

    auto inputType =
        dyn_cast<RankedTensorType>(broadcastOp.getInput().getType());
    if (!inputType)
      return failure();
    auto outputType =
        dyn_cast<RankedTensorType>(broadcastOp.getResult()[0].getType());
    // Get the positions of the input dimensions in the output.
    auto inputDimPositions = broadcastOp.getDimensions();
    auto isBroadcastedDimension = BitVector(outputType.getRank(), false);
    for (auto &dim : inputDimPositions) {
      isBroadcastedDimension[dim] = true;
    }

    // Get the offsets and sizes from the slice operation.
    auto outputOffsets = sliceOp.getMixedOffsets();
    auto outputSizes = sliceOp.getMixedSizes();

    // Compute the input offsets and sizes.
    SmallVector<OpFoldResult> inputOffsets;
    SmallVector<OpFoldResult> inputSizes;
    LLVM_DEBUG(llvm::dbgs() << broadcastOp << "\n";);
    for (int position = 0; position < outputType.getRank(); position++) {
      if (!isBroadcastedDimension[position]) {
        inputOffsets.push_back(outputOffsets[position]);
        inputSizes.push_back(outputSizes[position]);
      }
    }
    SmallVector<OpFoldResult> inputStrides(isBroadcastedDimension.size() -
                                               isBroadcastedDimension.count(),
                                           rewriter.getIndexAttr(1));

    // Create the extract_slice of the input.
    Location loc = broadcastOp.getLoc();
    rewriter.setInsertionPoint(broadcastOp);
    Value tiledInput = rewriter.create<tensor::ExtractSliceOp>(
        loc, broadcastOp.getInput(), inputOffsets, inputSizes, inputStrides);
    LLVM_DEBUG(llvm::dbgs() << tiledInput << "\n";);

    Value tiledInit = rewriter.create<tensor::ExtractSliceOp>(
        loc, broadcastOp.getInit(), sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes(), sliceOp.getMixedStrides());

    // Create the new BroadcastOp with the tiled input.
    SmallVector<Value> newOperands = {tiledInput, tiledInit};
    rewriter.setInsertionPointAfter(broadcastOp);
    Operation *newOp =
        clone(rewriter, broadcastOp, {sliceOp.getType()}, newOperands);
    rewriter.replaceOp(sliceOp, newOp->getResults());
    return success();
  }

  LogicalResult bubbleUpReduce(tensor::ExtractSliceOp sliceOp,
                               PatternRewriter &rewriter) const {
    auto reduceOp =
        dyn_cast<linalg::ReduceOp>(sliceOp.getSource().getDefiningOp());
    // For reduce, the output has lower rank than the input.
    // We need to map the output slice back to the input dimensions.

    // Get the reduction dimensions.
    ArrayRef<int64_t> reductionDims = reduceOp.getDimensions();

    // Build a map of reduction dimensions.
    auto inputType =
        dyn_cast<RankedTensorType>(reduceOp.getInputs()[0].getType());
    if (!inputType)
      return failure();
    auto inputRank = inputType.getRank();
    BitVector isReductionDim(inputRank, false);
    for (int64_t dim : reductionDims) {
      isReductionDim[dim] = true;
    }

    // Get the offsets and sizes from the slice operation.
    auto outputOffsets = sliceOp.getMixedOffsets();
    auto outputSizes = sliceOp.getMixedSizes();

    // Compute the input offsets and sizes.
    // Assert unit stride
    SmallVector<OpFoldResult> inputStrides(inputRank, rewriter.getIndexAttr(1));
    SmallVector<Value> newDpsInputs;
    SmallVector<Value> newDpsInits;
    for (int i = 0; i < reduceOp.getNumDpsInits(); i++) {
      unsigned outIdx = 0;
      SmallVector<OpFoldResult> inputOffsets(inputRank);
      SmallVector<OpFoldResult> inputSizes(inputRank);
      auto inputReduce = reduceOp.getDpsInputOperand(i)->get();
      auto initReduce = reduceOp.getDpsInitOperand(i)->get();
      auto mixedSizeFinal =
          tensor::getMixedSizes(rewriter, reduceOp.getLoc(), inputReduce);
      for (unsigned inIdx = 0; inIdx < inputRank; ++inIdx) {
        if (isReductionDim[inIdx]) {
          inputOffsets[inIdx] = rewriter.getIndexAttr(0);
          if (inputType.isDynamicDim(inIdx)) {
            inputSizes[inIdx] = mixedSizeFinal[inIdx].get<Value>();
          } else {
            inputSizes[inIdx] =
                rewriter.getIndexAttr(inputType.getDimSize(inIdx));
          }
        } else {
          inputOffsets[inIdx] = outputOffsets[outIdx];
          inputSizes[inIdx] = outputSizes[outIdx];
          ++outIdx;
        }
      }
      rewriter.setInsertionPoint(reduceOp);
      Value tiledInput = rewriter.create<tensor::ExtractSliceOp>(
          inputReduce.getLoc(), inputReduce, inputOffsets, inputSizes,
          inputStrides);
      Value tiledInit = rewriter.create<tensor::ExtractSliceOp>(
          initReduce.getLoc(), initReduce, outputOffsets, outputSizes,
          sliceOp.getMixedStrides());
      newDpsInputs.push_back(tiledInput);
      newDpsInits.push_back(tiledInit);
    }

    // Create the new ReduceOp with tiled operands.
    SmallVector<Value> newOperands;
    newOperands.append(newDpsInputs.begin(), newDpsInputs.end());
    newOperands.append(newDpsInits.begin(), newDpsInits.end());
    Operation *newOp =
        clone(rewriter, reduceOp, ValueRange(newDpsInits), newOperands);

    rewriter.replaceOp(sliceOp, newOp->getResults());
    return success();
  }
#endif

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const final {
    Value source = sliceOp.getSource();
    auto linalgOp = source.getDefiningOp<LinalgOp>();
    if (!linalgOp) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "expected source to be linalg op");
    }

#if BSPUB_DAVINCI_BISHENGIR
    if (!linalgOp->hasOneUse() &&
        (!options.aggressive || tensor::isOffsetBytesAligned(sliceOp, 32))) {
      return rewriter.notifyMatchFailure(
          sliceOp, "expected single use of linalg op or aggressive bubble up "
                   "for unaligned extract slice");
    }
#else
    // TODO: we might relax this if we want heuristics to detect that all uses
    // are small portion of the output.
    if (!linalgOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "expected single use of linalg op");
    }
#endif

    if (linalgOp.getNumDpsInits() != 1) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "expected single output of linalg op");
    }

    if (!linalgOp.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "expected tensor of linalg op");
    }

    if (!sliceOp.hasUnitStride())
      return rewriter.notifyMatchFailure(sliceOp, "expected unit stride");

    if (sliceOp.getType().getRank() != sliceOp.getSourceType().getRank()) {
      return rewriter.notifyMatchFailure(sliceOp, "expected no rank reduction");
    }

#if BSPUB_DAVINCI_BISHENGIR
    if (isa<linalg::BroadcastOp>(linalgOp)) {
      return bubbleUpBroadcast(sliceOp, rewriter);
    }
    if (isa<linalg::ReduceOp>(linalgOp)) {
      return bubbleUpReduce(sliceOp, rewriter);
    }
#endif

    OpOperand *outOperand = linalgOp.getDpsInitOperand(0);
    AffineMap indexingMap = linalgOp.getMatchingIndexingMap(outOperand);
    if (!indexingMap.isProjectedPermutation()) {
      return rewriter.notifyMatchFailure(
          sliceOp, "expected a projected permutation for output");
    }

    auto linalgLoc = linalgOp.getLoc();
    SmallVector<OpFoldResult> allShapeSizes =
        linalgOp.createFlatListOfOperandDims(rewriter, linalgLoc);
    AffineMap shapeSizesToLoopsMap = linalgOp.getShapesToLoopsMap();
    if (!shapeSizesToLoopsMap) {
      return rewriter.notifyMatchFailure(
          linalgOp, "failed to get loops map from shape sizes");
    }
    SmallVector<OpFoldResult> sizeBounds =
        affine::makeComposedFoldedMultiResultAffineApply(
            rewriter, linalgLoc, shapeSizesToLoopsMap, allShapeSizes);

    // The offsets and sizes from the slice operation only give you the tile
    // size of the output. Use that compute the tile sizes and offsets of the
    // loops. For loops not used to access the output, set the tile sizes to
    // loop bounds and set the offset to 0.
    SmallVector<OpFoldResult> tileOffsets(sizeBounds.size(),
                                          rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> tileSizes = sizeBounds;
    for (auto const &result : enumerate(indexingMap.getResults())) {
      unsigned position = cast<AffineDimExpr>(result.value()).getPosition();
      tileOffsets[position] = sliceOp.getMixedOffsets()[result.index()];
      tileSizes[position] = sliceOp.getMixedSizes()[result.index()];
    }

    SmallVector<Value> valuesToTile = linalgOp->getOperands();
    SmallVector<Value> tiledOperands =
        makeTiledShapes(rewriter, linalgLoc, linalgOp, valuesToTile,
                        tileOffsets, tileSizes, sizeBounds,
                        /*omitPartialTileCheck=*/true);

    SmallVector<Type, 4> resultTensorTypes;
    for (OpOperand &opOperand : linalgOp.getDpsInitsMutable())
      resultTensorTypes.push_back(
          tiledOperands[opOperand.getOperandNumber()].getType());

    Operation *newOp =
        clone(rewriter, linalgOp, resultTensorTypes, tiledOperands);
    rewriter.replaceOp(sliceOp, newOp->getResults());
    return success();
  }

#if BSPUB_DAVINCI_BISHENGIR
private:
  BubbleUpExtractSliceOptions options;
#endif
};
} // namespace

#if BSPUB_DAVINCI_BISHENGIR
void mlir::linalg::populateBubbleUpExtractSliceOpPatterns(
    RewritePatternSet &patterns, const BubbleUpExtractSliceOptions &options) {
  auto *context = patterns.getContext();
  patterns.add<BubbleUpExtractSliceOpPattern>(context, options);
}
#else
void mlir::linalg::populateBubbleUpExtractSliceOpPatterns(
    RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<BubbleUpExtractSliceOpPattern>(context);
}
#endif
