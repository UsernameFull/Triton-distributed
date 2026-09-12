//===- Options.cpp - Option related classes -------------------------------===//
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

#include "bishengir/Tools/ConfigOptions/Options.h"
#include "mlir/TableGen/Pass.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace bishengir::tblgen;

namespace {

std::string convertToDashSnakeFromCamelCase(StringRef input) {
  auto snakeCaseStr = llvm::convertToSnakeFromCamelCase(input);
  std::replace(snakeCaseStr.begin(), snakeCaseStr.end(), '_', '-');
  return snakeCaseStr;
}

std::string lowerFirstLetter(StringRef input) {
  if (input.empty()) {
    return "";
  }
  std::string result = input.str();
  if (std::isalpha(result[0]))
    result[0] = std::tolower(result[0]);

  return result;
}

} // namespace

namespace bishengir::tblgen {

//===----------------------------------------------------------------------===//
// ConfigOptions
//===----------------------------------------------------------------------===//

ConfigOption::ConfigOption(const llvm::Record *def) : def(def) {
  StringRef defName = def->getName();
  this->cppName = lowerFirstLetter(defName);
  this->argument = convertToDashSnakeFromCamelCase(defName);
  this->externalStorageLocation = this->cppName + "Flag";
}

StringRef ConfigOption::getCppVariableName() const { return this->cppName; }

StringRef ConfigOption::getCapitalizedCppVariableName() const {
  return def->getName();
}

StringRef ConfigOption::getArgument() const { return this->argument; }

StringRef ConfigOption::getType() const {
  return def->getValueAsString(OptionFields::kType);
}

std::optional<StringRef> ConfigOption::getDefaultValue() const {
  StringRef defaultVal = def->getValueAsString(OptionFields::kDefaultValue);
  return defaultVal.empty() ? std::optional<StringRef>() : defaultVal;
}

StringRef ConfigOption::getDescription() const {
  return def->getValueAsString(OptionFields::kDescription);
}

std::optional<StringRef> ConfigOption::getAdditionalFlags() const {
  StringRef additionalFlags =
      def->getValueAsString(OptionFields::kAdditionalOptFlags);
  return additionalFlags.empty() ? std::optional<StringRef>() : additionalFlags;
}

bool ConfigOption::getExternalStorage() const {
  return def->getValueAsBit(OptionFields::kExternalStorage);
}

std::optional<StringRef> ConfigOption::getExternalStorageLocation() const {
  if (!getExternalStorage())
    return std::nullopt;

  return this->externalStorageLocation;
}

std::vector<StringRef> ConfigOption::getPassGroups() const {
  return def->getValueAsListOfStrings(OptionFields::kPassGroup);
}

std::vector<mlir::StringRef> ConfigOption::getOptionCategories() const {
  return def->getValueAsListOfStrings(OptionFields::kCompileOptionCategories);
}

std::optional<StringRef> ConfigOption::getCompileConfigName() const {
  return def->getValueAsOptionalString(OptionFields::kCompileConfigName);
}

bool ConfigOption::getEmitGetterSetter() const {
  return def->getValueAsBit(OptionFields::kEmitGetterSetter);
}

bool ConfigOption::getEmitOptionRegistration() const {
  return def->getValueAsBit(OptionFields::kEmitOptionRegistration);
}

bool ConfigOption::isListOption() const {
  return def->isSubClassOf(OptionFields::kListOption);
}

std::string getContainerType(const ConfigOption &opt) {
  // By default use vector to hold list data
  if (opt.isListOption())
    return "std::vector<" + opt.getType().str() + ">";

  return opt.getType().str();
}

} // namespace bishengir::tblgen
