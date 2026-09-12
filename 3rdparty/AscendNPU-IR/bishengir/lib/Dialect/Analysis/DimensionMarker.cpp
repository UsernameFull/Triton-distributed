//===- DimensionMarker.cpp ------------------------------------------------===//
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
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Utils/Util.h"

#include <numeric>

using namespace mlir;
using namespace mlir::utils::debugger;
using namespace mlir::reshape_utils;
using namespace mlir::tensor::reshape_utils;

#define DEBUG_TYPE "dimension-analyzer-marker"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace detail {

void DimensionAnalyzerBase::dumpModuleOP() const {
  LLVM_DEBUG(llvm::dbgs() << *op_ << "\n";);
}

bool ExtendedUnionFind::join(int a, int b) {
  assert(shape_[a] != kUndefinedShaped);
  assert(shape_[b] != kUndefinedShaped);
  assert(minParentIndex_.size() == parent_.size());
  assert(std::max(a, b) < parent_.size());
  a = find(a);
  b = find(b);
  if (a != b) {
    if (parent_[a] > parent_[b])
      std::swap(a, b);
    parent_[a] += parent_[b];
    minIndex[a] = std::min(minIndex[b], minIndex[a]);
    // Derive min parent index
    minParentIndex_[a] = std::min(minParentIndex_[a], minParentIndex_[b]);
    assert(minParentIndex_[a] != kMaxDimPos &&
           "Inconsistency found when assigning parent");
    // Derive shape
    if (shape_[a] != ShapedType::kDynamic &&
        shape_[b] != ShapedType::kDynamic) {
      // if both is static, then verify
      if (shape_[a] != shape_[b])
        return false;
    }
    if (shape_[b] != ShapedType::kDynamic) {
      // if b is static, then propagate to a
      shape_[a] = shape_[b];
    }
    parent_[b] = a;
  }
  return true;
}

void ExtendedUnionFind::allocateMinimum(size_t n) {
  UnionFindBase::allocateMinimum(n);
  if (n + 1 > shape_.size()) {
    shape_.resize(n + 1, kUndefinedShaped);
    minParentIndex_.resize(n + 1, kMaxDimPos);
  }
}

std::pair<DimensionPosition, int64_t>
ExtendedUnionFind::getMinParentAndShapePair(int a) {
  auto parent = find(a);
  return std::make_pair(minParentIndex_[parent], shape_[parent]);
}

/// This function tries to combine extra information for shapes, for example
/// tensor.empty and its tensor.dim components. This is useful for minimum
/// information in dynamic shape context
void DimensionAnalyzerBase::combineInferable() {
  for (const auto &arg : argumentList_) {
    combineEmptyOp(arg);
  }
}

void DimensionAnalyzerBase::combineEmptyOp(Value arg) {
  auto emptyOp = dyn_cast_if_present<tensor::EmptyOp>(arg.getDefiningOp());
  if (!emptyOp) {
    return;
  }

  LDBG("Combining empty op " << emptyOp);
  auto emptyRef = getArgumentRef(emptyOp.getResult());
  auto mixEmptyShape = emptyOp.getMixedSizes();

  for (auto [emptyIdx, el] : llvm::enumerate(mixEmptyShape)) {
    // Skip constant values
    if (getConstantIntValue(el).has_value()) {
      return;
    }

    auto sizeDefiner = cast<Value>(el);
    auto dimOp =
        dyn_cast_if_present<tensor::DimOp>(sizeDefiner.getDefiningOp());
    if (!dimOp) {
      return;
    }

    LDBG("Found dim op " << dimOp);
    linkDimToEmpty(dimOp, emptyRef[emptyIdx]);
  }
}

void DimensionAnalyzerBase::linkDimToEmpty(tensor::DimOp dimOp,
                                           int64_t emptyRefElement) {
  auto constantIndex = dimOp.getConstantIndex();
  if (!constantIndex.has_value()) {
    return;
  }
  // Check if the option is on
  if (!bindUsingTensorDim) {
    return;
  }
  auto tensorSource = dimOp.getSource();
  auto tensorRef = getArgumentRefOrCreateDummy(tensorSource);
  joinShape(tensorRef[constantIndex.value()], emptyRefElement);
}

/// \param values this loops through all value and will create new terminal
/// on-the-fly
void DimensionAnalyzerBase::createDummyRefIfNotExist(ArrayRef<Value> values) {
  for (auto curVal : values) {
    if (argumentsRefPointer_.contains(curVal))
      continue;
    LDBG("Creating dummy for value: " << curVal);
    // init elements
    auto [rank, shape] = utils::getValueShapeInfo(curVal).value_or(
        std::make_pair(0, DimensionShape{}));
    int startingIdx = allocateArguments(rank, shape);
    argumentsRef_.push_back(DimensionShape(shape));
    std::iota(argumentsRef_.back().begin(), argumentsRef_.back().end(),
              startingIdx);
    initCollapseOrVerify(curVal,
                         static_cast<int64_t>(argumentsRef_.size() - 1));
  }
}

void DimensionAnalyzerBase::updatePreviousType(const Value &val) {
  RankedTensorType curType = dyn_cast<RankedTensorType>(val.getType());
  if (curType)
    updatePreviousType(val, curType);
  else {
    LLVM_DEBUG(llvm::dbgs() << val << " is not inittable\n";);
  }
}

void DimensionAnalyzerBase::updatePreviousType(
    const Value &val, const RankedTensorType &curType) {
  if (previousType_.contains(val)) {
    assert(previousType_[val] == curType);
  }
  previousType_[val] = curType;
}

// Mark all value that is the result of collapse
void DimensionAnalyzerBase::collapsePropagateOrVerify(Operation *op,
                                                      const Value &refVal) {
  const auto tmpVal = argumentsRefPointer_.at(refVal);
  for (const Value newVal : op->getResults()) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Propagating " << newVal << " " << tmpVal << "\n";);
    if (argumentsRefPointer_.contains(newVal)) {
      solverSegments_->join(argumentsRefPointer_.at(newVal), tmpVal);
      return;
    }
    argumentsRefPointer_[newVal] = tmpVal;
  }
}

void DimensionAnalyzerBase::collapsePropagateOrVerify(const Value &newVal,
                                                      const Value &arg) {
  const auto tmpVal = argumentsRefPointer_.at(arg);
  LLVM_DEBUG(llvm::dbgs() << "Propagating " << newVal << " " << tmpVal
                          << "\n";);
  if (argumentsRefPointer_.contains(newVal)) {
    solverSegments_->join(argumentsRefPointer_.at(newVal), tmpVal);
    return;
  }
  argumentsRefPointer_[newVal] = tmpVal;
}

void DimensionAnalyzerBase::initCollapseOrVerify(const Value &val,
                                                 int64_t refPtr) {
  LLVM_DEBUG(llvm::dbgs() << "Assigning " << val << " " << refPtr << "\n";);
  if (argumentsRefPointer_.contains(val)) {
    LLVM_DEBUG(llvm::dbgs() << "Init has been done previously\n";);
    solverSegments_->join(argumentsRefPointer_.at(val), refPtr);
    return;
  }
  argumentsRefPointer_[val] = refPtr;
}

// Step 3: Unifying groups of segments
void DimensionAnalyzerBase::unifyGroups() {
  for (int ref = 0; ref < static_cast<int>(argumentsRef_.size()); ++ref) {
    LLVM_DEBUG(llvm::dbgs() << "\nUnifying ---> " << ref << " \n";);
    auto parIdx = solverSegments_->find(ref);
    if (parIdx == ref)
      continue;

    const auto &par = argumentsRef_[parIdx];
    LDBG("Parent index (" << parIdx << ") - child Index(" << ref << "): "
                          << par.size() << " " << argumentsRef_[ref].size());
    assert(par.size() == argumentsRef_[ref].size());
    for (const auto &[idx, u] : llvm::enumerate(argumentsRef_[ref])) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Unifying " << u << " with " << par[idx] << "\n");
      joinShape(u, par[idx]);
    }
  }
}

void DimensionAnalyzerBase::propagateConnection(int parent, int child) {
  isConnected_[parent].leftConnected =
      isConnected_[parent].leftConnected && isConnected_[child].leftConnected;
  isConnected_[parent].rightConnected =
      isConnected_[parent].rightConnected && isConnected_[child].rightConnected;
  LDBG(static_cast<int64_t>(isConnected_[child].elementKind)
       << " is propagated from " << child << " to " << parent);
  isConnected_[parent].elementKind = std::min(isConnected_[parent].elementKind,
                                              isConnected_[child].elementKind);
}

void DimensionAnalyzerBase::propagateConnection() {
  for (int i = 0; i < argumentTotalLength_; ++i) {
    // Use collapser because the relationship is stronger
    int parent = solverCollapserElem_->find(i);
    propagateConnection(parent, i);
  }
}

void DimensionAnalyzerBase::spreadConnection() {
  for (int i = 0; i < argumentTotalLength_; ++i) {
    isConnected_[i] = isConnected_[solverCollapserElem_->find(i)];
    // check if this is available in the arguments
    LLVM_DEBUG(llvm::dbgs()
                   << i << " Found shape is "
                   << solverShapeElem_->getMinParentAndShapePair(i).second
                   << " parent is " << solverShapeElem_->find(i) << " "
                   << isConnected_[i].leftConnected << " "
                   << isConnected_[i].rightConnected << "\n";);
  }
}

bool DimensionAnalyzerBase::isConnected(int a, int b) {
  if (a + 1 != b)
    return false;
  return isConnected_[a].rightConnected && isConnected_[b].leftConnected;
}

void DimensionAnalyzerBase::joinShape(int a, int b) {
  LDBG("Joining shape bind " << a << " " << b);
  solverShapeElem_->join(a, b);
  solverCollapserElem_->join(a, b);
}

void DimensionAnalyzerBase::joinCollapser(int a, int b) {
  LDBG("Joining collapser bind " << a << " " << b);
  solverCollapserElem_->join(a, b);
}

void DimensionAnalyzerBase::disconnect(int a, int b) {
  if (0 <= a && a < static_cast<int>(isConnected_.size())) {
    isConnected_[a].rightConnected = false;
    int parentOfA = solverCollapserElem_->find(a);
    isConnected_[parentOfA].rightConnected = false;
  }
  if (0 <= b && b < static_cast<int>(isConnected_.size())) {
    isConnected_[b].leftConnected = false;
    int parentOfB = solverCollapserElem_->find(b);
    isConnected_[parentOfB].leftConnected = false;
  }
}

} // namespace detail
} // namespace mlir
