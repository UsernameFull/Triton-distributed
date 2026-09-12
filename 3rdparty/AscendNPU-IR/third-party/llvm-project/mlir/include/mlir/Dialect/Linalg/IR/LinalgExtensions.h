//===- LinalgExtensions.h - Linalg operations extensions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation extensions for Linalg operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_IR_LINALGEXTENSIONS_H_
#define MLIR_DIALECT_LINALG_IR_LINALGEXTENSIONS_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace linalg {

/// Returns true if value is splat dense, i.e. only has one single element.
bool isSplatDense(Value value);

/// Create new constant op from splat dense value content.
std::optional<Value> createConstantFromDenseSplat(Value value,
                                                  PatternRewriter &rewriter);

void cloneRegion(Operation *op, IRMapping &mapping, PatternRewriter &rewriter);

template <typename OpType>
struct SimplifySplatDenseForBinary : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  SimplifySplatDenseForBinary(MLIRContext *context)
      : OpRewritePattern<OpType>(context, /*benefit=*/1) {} // Lower benefit

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getDpsInputs();
    if (inputs.size() != 2) {
      return rewriter.notifyMatchFailure(op, "invalid dps input num");
    }

    Value lhs = inputs[0];
    Value rhs = inputs[1];
    if (!isSplatDense(lhs) && !isSplatDense(rhs)) {
      return rewriter.notifyMatchFailure(op, "no splat dense operand found");
    }

    if (lhs.getType().isIntOrIndexOrFloat() ||
        rhs.getType().isIntOrIndexOrFloat()) {
      return rewriter.notifyMatchFailure(op, "already has scalar in operand");
    }

    unsigned int operIdx = (isSplatDense(lhs) ? 0 : 1);
    OpOperand &oper = op->getOpOperand(operIdx);
    auto scalarMaybe = createConstantFromDenseSplat(oper.get(), rewriter);
    if (!scalarMaybe.has_value()) {
      return rewriter.notifyMatchFailure(op, "failed to get dense constant.");
    }
    Value scalar = scalarMaybe.value();
    oper.set(scalar);
    return success();
  }
};

template <typename OpType>
struct InlineDenseSplatToGenericRegion : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  /// Inline Dense Splat Constants to Generic Region
  ///
  /// For example:
  /// ```mlir
  ///    linalg.elemwise_unary {fun = #linalg.unary_fn<negf>}
  ///             ins(%constant : tensor<16x16xf32>)
  /// ```
  ///
  /// Will be converted to:
  /// ```mlir
  ///    arith.negf %constant : tensor<16x16xf32>
  /// ```
  InlineDenseSplatToGenericRegion(MLIRContext *context)
      : OpRewritePattern<OpType>(context, /*benefit=*/2) {}
  // For linalg/hfusion binary ops this conflicts with
  // SimplifySplatDenseForBinary We give this pattern higher benefit since it's
  // better to inline to generic region than changing one of the operands to a
  // scalar
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {

    bool inputSplatDense = llvm::all_of(
        op.getDpsInputs(), [](Value input) { return isSplatDense(input); });

    bool outputEmptyTensor = llvm::all_of(op.getDpsInits(), [](Value init) {
      Operation *defOp = init.getDefiningOp();
      auto emptyTensor = dyn_cast_or_null<tensor::EmptyOp>(defOp);
      return static_cast<bool>(emptyTensor);
    });

    if (!outputEmptyTensor)
      return rewriter.notifyMatchFailure(op, "output is not empty tensor");

    if (!inputSplatDense)
      return rewriter.notifyMatchFailure(op,
                                         "input is not dense splat/constant");

    // Step 1. Set insert pos
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    // Step 2. Map arguments of block to op input/result.
    Block &block = op->getRegions().front().getBlocks().front();
    IRMapping mapping;
    auto inputs = op.getDpsInputs();
    auto results = op.getDpsInits();
#ifndef NDEBUG
    auto arguments = block.getArguments();
    assert(arguments.size() == inputs.size() + results.size());
#endif
    int cnt = 0;
    for (Value in : inputs) {
      mapping.map(block.getArgument(cnt++), in);
    }
    for (Value res : results) {
      mapping.map(block.getArgument(cnt++), res);
    }

    // Checking for generic region. might not be needed as we dont support
    // Load/Store anymore
    if (std::distance(block.getOperations().begin(),
                      block.getOperations().end()) == 1) {
      auto singleOp = block.getOperations().begin();
      if (singleOp->hasTrait<OpTrait::IsTerminator>()) {
        return rewriter.notifyMatchFailure(op, "generic region is empty");
      }
    }

    // Step 3. Travese operations in block and convert scalar into shaped
    // tensor.
    cloneRegion(op, mapping, rewriter);

    // Step 4. replace res with yieldop res.
    auto *terminator = block.getTerminator();
    assert(terminator);
    assert(isa<linalg::YieldOp>(terminator));
    auto yieldOp = cast<linalg::YieldOp>(terminator);
    for (auto [res, yieldOper] :
         llvm::zip(op->getResults(), yieldOp.getOperands())) {
      rewriter.replaceAllUsesWith(res, mapping.lookup(yieldOper));
    }

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename OpType>
struct RefactorRedundantReduceLikeOp : OpRewritePattern<OpType> {
public:
  // move out operations inside operation for unchanging shape reduce op
  // example:
  //
  // linalg.reduce ins(%arg0) outs(%arg1) dimensions = []
  //   (%in, %init) {
  //     %0 = arith.addf %in, %init
  //     %1 = arith.mulf %in, %0
  //     linalg.yield %1
  //   }
  //
  // to
  //
  // %0 = arith.addf %arg0, %arg1
  // %1 = arith.mulf %arg0, %0

  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType reduceOp,
                                PatternRewriter &rewriter) const override {
    if (!reduceOp.getDimensions().empty())
      return rewriter.notifyMatchFailure(
          reduceOp, "input and init should have same shape");

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(reduceOp);

    Block &block = reduceOp->getRegions().front().getBlocks().front();

    // create mapping from block argument to input/inits
    IRMapping newOperandsMap;
    auto inputs = reduceOp.getDpsInputs();
    auto results = reduceOp.getDpsInits();
#ifndef NDEBUG
    auto arguments = block.getArguments();
    assert(arguments.size() == inputs.size() + results.size());
#endif
    int cnt = 0;
    for (Value in : inputs) {
      newOperandsMap.map(block.getArgument(cnt++), in);
    }
    for (Value res : results) {
      newOperandsMap.map(block.getArgument(cnt++), res);
    }

    // note: we will infer the return types for every cloned operations
    cloneRegion(reduceOp, newOperandsMap, rewriter);

    // get replacement from mapping and do replacement
    Operation *terminator = block.getTerminator();
    assert(terminator);
    assert(isa<linalg::YieldOp>(terminator));
    linalg::YieldOp yieldOp = cast<linalg::YieldOp>(terminator);
    for (auto [res, yieldOpr] :
         llvm::zip(reduceOp->getResults(), yieldOp->getOperands())) {
      assert(newOperandsMap.contains(yieldOpr) && "Operand is not mapped");
      rewriter.replaceAllUsesWith(res, newOperandsMap.lookup(yieldOpr));
    }

    rewriter.eraseOp(reduceOp);
    return success();
  }
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_IR_LINALGEXTENSIONS_H_
