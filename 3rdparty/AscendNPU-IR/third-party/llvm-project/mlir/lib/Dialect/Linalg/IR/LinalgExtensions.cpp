//===- LinalgExtensions.cpp - Linalg operations extensions ----------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation extensions for Linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgExtensions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Linalg Operations Extensions
//===----------------------------------------------------------------------===//

bool mlir::linalg::isSplatDense(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
  auto constantOp = dyn_cast_or_null<arith::ConstantOp>(defOp);
  if (!constantOp) {
    return false;
  }
  auto constValue = constantOp.getValue();
  auto denseAttr = dyn_cast_or_null<DenseIntOrFPElementsAttr>(constValue);
  if (!denseAttr || !denseAttr.isSplat()) {
    return false;
  }
  auto elemType = denseAttr.getElementType();
  return elemType.isIntOrIndexOrFloat();
}

std::optional<Value>
mlir::linalg::createConstantFromDenseSplat(Value value,
                                           PatternRewriter &rewriter) {
  if (!isSplatDense(value)) {
    return std::nullopt;
  }
  auto op = cast<arith::ConstantOp>(value.getDefiningOp());
  auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(op.getValue());
  if (!denseAttr || !denseAttr.isSplat()) {
    return std::nullopt;
  }
  Type elemType = denseAttr.getElementType();
  Location loc = op.getLoc();
  TypedAttr attr;
  if (elemType.isInteger()) {
    APInt value = denseAttr.getSplatValue<APInt>();
    attr = rewriter.getIntegerAttr(elemType, value);
  } else if (isa<FloatType>(elemType)) {
    APFloat value = denseAttr.getSplatValue<APFloat>();
    attr = rewriter.getFloatAttr(elemType, value);
  } else {
    return std::nullopt;
  }
  return rewriter.create<arith::ConstantOp>(loc, elemType, attr);
}

void mlir::linalg::cloneRegion(Operation *opWithRegion, IRMapping &mapping,
                               PatternRewriter &rewriter) {
  for (Region &region : opWithRegion->getRegions()) {
    for (Block &block : region.getBlocks()) {
      block.walk([&](Operation *op) {
        if (op->hasTrait<OpTrait::IsTerminator>())
          return;

        Operation *newOp = rewriter.clone(*op, mapping);

        // set operands new types
        // to convert scalar into shaped tensor for operation like
        // arith.sitofp.
        // For example,
        // %1 = "arith.sitofp"(%arg0) : (i64) -> bf16
        // that should be
        // %1 = arith.sitofp %arg0 : tensor<i64> to tensor<bf16>
        for (auto [i, resType] :
             llvm::enumerate(opWithRegion->getResultTypes())) {
          auto currentOpResShape = resType.cast<ShapedType>().getShape();

          for (auto [newRes, oldRes] :
               llvm::zip(newOp->getResults(), op->getResults())) {
            auto newType = RankedTensorType::get(currentOpResShape,
                                                 getElementTypeOrSelf(newRes));
            newRes.setType(newType);
            mapping.map(oldRes, newRes);
          }
        }
      });
    }
  }
}
