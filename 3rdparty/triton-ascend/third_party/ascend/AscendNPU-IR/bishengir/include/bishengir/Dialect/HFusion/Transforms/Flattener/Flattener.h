//===- Flattener.h --------------------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FLATTENER_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FLATTENER_H

#include "bishengir/Dialect/HFusion/Analysis/DimensionAnalyzer.h"

namespace mlir {
namespace hfusion {
namespace detail {

class Flattener : public DimensionAnalyzer {
public:
  /// Constructor that takes the operation to flatten.
  Flattener(Operation *op);

public:
  using CollapseGroup = SmallVector<ReassociationIndices>;

  /// Performs the flattening transformation on the operation.
  LogicalResult flatten(bool multiDynamicShape);

  /// Gets the collapse group for a given value.
private:
  /// Marks broken connections in dimensions.
  void markBroken(const DimensionIndex &args);

  /// Propagates broken connections.
  void propagateBroken();
  void breakDynamicShapes();

  bool computeMutation(int pos, int dir) const;

  CollapseGroup getCollapseGroup(Value res) const;

  bool hasCollapseGroup(Value res) const;

  /// Gets the flattened mixed sizes for a value.
  SmallVector<OpFoldResult> getFlattenMixedSizes(Value res) const;

  /// Adjusts operations after flattening.
  void adjustOperations();

  LogicalResult VerifyCollapsedOperand(Operation *op) const;

  /// Collapses block arguments.
  void collapserForArg(Value &arg, OpBuilder &builder);

  /// Adjusts collapse dimensions for certain operations.
  template <class T>
  SmallVector<int64_t> adjustCollapseDimensions(T op,
                                                CollapseGroup indices) const;

  /// Adjusts indices for tensor.extract operations.
  void adjustExtractOpIndices(tensor::ExtractOp extractOp, OpBuilder &builder);

  /// Adjusts indices for tensor.pad operations.
  void adjustPadOp(tensor::PadOp padOp, OpBuilder &builder);

  /// Adjusts indices for hfusion.gather operations.
  void adjustGatherOp(hfusion::GatherOp gatherOp, OpBuilder &builder);

  /// Adjusts indices for tensor.concat operations.
  void adjustConcatOp(tensor::ConcatOp concatOp);

  void adjustInterleaveOp(hfusion::InterleaveOp interleaveOp);

  void adjustDeinterleaveOp(hfusion::DeinterleaveOp deinterleaveOp);

  void adjustResultType(DestinationStyleOpInterface dpsLikeOp);

  void adjustBroadcastOp(linalg::BroadcastOp broadcastOp, OpBuilder &builder);

  void adjustTransposeOp(linalg::TransposeOp transposeOp,
                         OpBuilder &builder) const;

  /// Adjusts dimensions for reduce-like operations.
  template <class T> void adjustReduceLikeOpBody(T reduceOp) const;

  /// Adjusts dimensions for cumulative operations.
  template <class T> void adjustCumOp(T cumOp, OpBuilder &builder);

  template <class T>
  void computeNewSlicingOperands(T slicingOp,
                                 SmallVector<OpFoldResult> &newMixedOffsets,
                                 SmallVector<OpFoldResult> &newMixedSizes,
                                 SmallVector<OpFoldResult> &newMixedStrides,
                                 OpBuilder &builder);

  void adjustExtractSliceOp(tensor::ExtractSliceOp extractSliceOp,
                            OpBuilder &builder);

  void adjustInsertSliceOp(tensor::InsertSliceOp insertSliceOp,
                           OpBuilder &builder);

  /// Collapses operations during adjustment.
  LogicalResult collapser(Operation *op, OpBuilder &builder);

  /// Adjusts return operations after flattening.
  void adjustReturnOp(Operation *op, OpBuilder &builder) const;

  /// Collapses values for output operations.
  template <typename OpTy>
  FailureOr<tensor::ExpandShapeOp>
  collapseForOut(OpTy &tensorOutOp, Value &collapsedVal, OpBuilder &builder);

  /// Adjusts tensor output operations (source).
  template <typename OpTy>
  void adjustTensorOutOpSource(OpTy tensorOutOp, OpBuilder &builder);

  /// Adjusts tensor output operations (destination).
  template <typename OpTy>
  void adjustTensorOutOpDest(OpTy tensorOutOp, OpBuilder &builder);

  /// Adjusts tensor output operations (source alternative).
  template <typename OpTy>
  void adjustTensorOutOpSrc(OpTy tensorOutOp, OpBuilder &builder);

  void eraseOp(Operation *op);
  SetVector<Operation *> flattenerWorkList;
};

} // namespace detail
} // namespace hfusion
} // namespace mlir
#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FLATTENER_H