//===- Initializer.cpp ----------------------------------------------------===//
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

#include "bishengir/Dialect/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include <numeric>

using namespace mlir;
using namespace mlir::utils::debugger;

#define DEBUG_TYPE "dimension-analyzer-initialize"
namespace mlir {
namespace detail {

DimensionAnalyzerBase::DimensionAnalyzerBase(Operation *op) : op_(op) {}

LogicalResult DimensionAnalyzerBase::initialize() {
  // Check if tensor::ExpandShapeOp exists in the function
  bool hasReshaping = false;
  bool hasFunctionCall = false;
  op_->walk([&](Operation *op) {
    if (isa<func::CallOp>(op)) {
      hasFunctionCall = true;
    }
    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::ReshapeOp>(
            op)) {
      hasReshaping = true;
    }
    return WalkResult::advance();
  });
  if (hasReshaping) {
    LLVM_DEBUG(llvm::dbgs() << "Will try to optimize scoped reshape\n";);
  }
  if (hasFunctionCall) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Skipping function with function call inside\n";);
    return failure();
  }
  initializeStructures();
  processBFS();
  unifyGroups();
  return success();
}

int64_t
DimensionAnalyzerBase::allocateArguments(int rank,
                                         ArrayRef<int64_t> dimensionRef) {
  LLVM_DEBUG(llvm::dbgs() << "Allocating new arguments " << rank << "\n");
  auto startingIdx = argumentTotalLength_;
  argumentTotalLength_ += rank + 1;
  isConnected_.resize(argumentTotalLength_);
  solverShapeElem_->allocateMinimum(argumentTotalLength_);
  solverCollapserElem_->allocateMinimum(argumentTotalLength_);
  assert(rank == dimensionRef.size());
  for (int64_t i = 0; i < rank; ++i) {
    LLVM_DEBUG(llvm::dbgs()
                   << "allocating " << i << ": " << dimensionAllocation_ << " "
                   << dimensionRef[i] << "\n";);
    int64_t currentIndex = startingIdx + i;
    solverShapeElem_->minParentIndex_[currentIndex] = {dimensionAllocation_, i};
    solverShapeElem_->shape_[currentIndex] = dimensionRef[i];
    isConnected_[currentIndex].elementKind =
        dimensionRef[i] == 1 ? tensor::reshape_utils::ElementKind::Unit
                             : tensor::reshape_utils::ElementKind::NoMutation;
    if (i > 0)
      isConnected_[currentIndex].leftConnected = true;
    if (i + 1 < rank)
      isConnected_[currentIndex].rightConnected = true;
  }
  dimensionAllocation_++;

  LLVM_DEBUG(llvm::dbgs() << "Starting index: " << startingIdx << "\n";);
  return startingIdx;
}

// Step 1: Initializing arguments segments
void DimensionAnalyzerBase::initializeStructures() {
  solverShapeElem_ = std::make_unique<ExtendedUnionFind>();
  solverCollapserElem_ = std::make_unique<SimpleUnionFind>();
  solverSegments_ = std::make_unique<SimpleUnionFind>();

  size_t sizeCount = 0;
  for (Block &block : op_->getRegion(0)) {
    LLVM_DEBUG(llvm::dbgs() << "Processing Block\n");
    sizeCount += block.getOperations().size();

    // FLATTEN-IN
    // Process block arguments
    for (BlockArgument arg : block.getArguments()) {
      if (isa<TensorType>(arg.getType())) {
        processArgument(arg);
      }
    }

    // Process args of some knowing operations as an opener
    // operations
    block.walk([&](Operation *op) {
      if (reshape_utils::isArgOp(op)) {
        Value result = op->getResult(0);
        if (isa<TensorType>(result.getType())) {
          LLVM_DEBUG(llvm::dbgs() << "Putting " << result << " in arguments "
                                  << "\n";);
          processArgument(result);
        }
      }
    });
    block.walk([&](Operation *op) {
      if (reshape_utils::isOutOp(op)) {
        outList_.push_back(op);
      }
    });
  }

  LLVM_DEBUG(llvm::dbgs() << "Initializing structures sizeCount: " << sizeCount
                          << "\n");
  solverSegments_->allocateMinimum(sizeCount);
  assert(dimensionAllocation_ == argumentList_.size() &&
         "Inconsistency in argumentList_");
}

void DimensionAnalyzerBase::processArgument(Value arg) {
  OpBuilder builder(op_);
  argumentList_.push_back(arg);

  auto [rank, shape] = utils::getValueShapeInfo(arg).value_or(
      std::make_pair(0, DimensionShape{}));
  // Add size for space as well
  LLVM_DEBUG(llvm::dbgs() << "Found args: " << arg << ' ' << rank << "\n");
  auto startingIdx = allocateArguments(rank, shape);
  initCollapseOrVerify(arg, argumentsRef_.size());
  argumentsRef_.push_back(DimensionShape(shape));
  std::iota(argumentsRef_.back().begin(), argumentsRef_.back().end(),
            startingIdx);
  LLVM_DEBUG(llvm::dbgs() << utils::debugger::to_string(argumentsRef_.back())
                          << '\n');

  // args:
  // [2x4xf32]_[5xf32 ]_[8x7x6xf32]_
  //  0 1     2 3      4 5 6 7     8
  //
  //  2, 4 and 8 are spacing,
  //  each argument shape is assigned with an index
  //
  //  argumentsRefPointer_ : {arg0 : 0, arg1 : 1, arg2 : 2}
  //  argumentsRef_ : {{0,1}, {3}, {5,6,7}}
  //
  //  Broadcasting new elements will also increase the arguments ref,
  //  and create a new arguments ref pointer index
}
} // namespace detail
} // namespace mlir
