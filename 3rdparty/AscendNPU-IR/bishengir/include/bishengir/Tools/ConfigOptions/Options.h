//===- Options.h - TableGen Option definitions ------------------*- C++ -*-===//
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
#ifndef BISHENGIR_TOOLS_CONFIGOPTIONS_OPTIONS_H
#define BISHENGIR_TOOLS_CONFIGOPTIONS_OPTIONS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace llvm {
class Record;
} // namespace llvm

namespace bishengir {
namespace tblgen {

constexpr llvm::StringLiteral kOptionClassName = "Option";
namespace OptionFields {
constexpr llvm::StringLiteral kType = "type";
constexpr llvm::StringLiteral kDefaultValue = "defaultValue";
constexpr llvm::StringLiteral kDescription = "description";
constexpr llvm::StringLiteral kAdditionalOptFlags = "additionalOptFlags";
constexpr llvm::StringLiteral kExternalStorage = "externalStorage";
constexpr llvm::StringLiteral kPassGroup = "passGroup";
constexpr llvm::StringLiteral kCompileOptionCategories =
    "compileOptionCategories";
constexpr llvm::StringLiteral kCompileConfigName = "compileConfigName";
constexpr llvm::StringLiteral kEmitGetterSetter = "emitGetterSetter";
constexpr llvm::StringLiteral kListOption = "ListOption";
constexpr llvm::StringLiteral kEmitOptionRegistration =
    "emitOptionRegistration";
} // namespace OptionFields

class ConfigOption {
public:
  explicit ConfigOption(const llvm::Record *def);

  /// Get the original record.
  const llvm::Record *getDef() const { return def; }

  /// Return the name for the C++ option variable.
  mlir::StringRef getCppVariableName() const;

  /// Return the name for the capitalized C++ option variable.
  mlir::StringRef getCapitalizedCppVariableName() const;

  /// Return the command line argument to use for this option.
  mlir::StringRef getArgument() const;

  /// Return the C++ type of the option.
  mlir::StringRef getType() const;

  /// Return the default value of the option.
  std::optional<mlir::StringRef> getDefaultValue() const;

  /// Return the description for this option.
  mlir::StringRef getDescription() const;

  /// Return the additional flags passed to the option constructor.
  std::optional<mlir::StringRef> getAdditionalFlags() const;

  /// Return whether the config option has external storage.
  bool getExternalStorage() const;

  /// Return the external storage location for the config option.
  std::optional<mlir::StringRef> getExternalStorageLocation() const;

  /// Return the list of pass pipeline groups that this option belongs to.
  std::vector<mlir::StringRef> getPassGroups() const;

  /// Return the list of categories that this option belongs to.
  std::vector<mlir::StringRef> getOptionCategories() const;

  /// Returns the compile config's class name.
  std::optional<mlir::StringRef> getCompileConfigName() const;

  /// Returns whether to emit getter or setter methods.
  bool getEmitGetterSetter() const;

  /// Returns whether to emit command line option registration.
  bool getEmitOptionRegistration() const;

  /// Flag indicating if this is a list option.
  bool isListOption() const;

private:
  const llvm::Record *def;

  std::string cppName{};
  std::string argument{};
  std::string externalStorageLocation{};
};

/// Get the cpp container type for the config variable.
/// If the config is a list option, the container type is
///   `std::vector<Type>`
/// Otherwise, it's just `Type`.
std::string getContainerType(const ConfigOption &opt);

} // namespace tblgen
} // namespace bishengir

#endif // BISHENGIR_TOOLS_CONFIGOPTIONS_OPTIONS_H