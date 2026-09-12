//===- PassManager.h - Pass Management Interface ----------------*- C++ -*-===//
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

#ifndef BISHENGIR_PASS_PASSMANAGER_H
#define BISHENGIR_PASS_PASSMANAGER_H

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Tools/BiShengIRConfigBase/Config.h"
#include "mlir/Pass/PassManager.h"

namespace bishengir {
/// Register a set of useful command-line options that can be used to configure
/// a pass manager. The values of these options can be applied via the
/// 'applyPassManagerCLOptions' method below.
void registerPassManagerCLOptions();

/// Apply any values provided to the pass manager options that were registered
/// with 'registerPassManagerOptions'.
llvm::LogicalResult applyPassManagerCLOptions(mlir::PassManager &pm);

// A pass manager that allows filtering the passes before running. It's more
// expensive to use with compared to mlir::PassManager.
class BiShengIRPassManager : public mlir::PassManager {
public:
  using PassManager::PassManager;
  BiShengIRCompileConfigBase config;

  BiShengIRPassManager(const BiShengIRCompileConfigBase &config,
                       mlir::MLIRContext *ctx, llvm::StringRef operationName,
                       Nesting nesting)
      : PassManager(ctx, operationName, nesting), config(config){};

#if MLIR_ENABLE_EXECUTION_ENGINE
  mlir::LogicalResult run(mlir::Operation *op);

private:
  void filterCPURunnerPasses(mlir::OpPassManager &originalPM);
#endif // MLIR_ENABLE_EXECUTION_ENGINE
};

} // namespace bishengir

#endif // BISHENGIR_PASS_PASSMANAGER_H
