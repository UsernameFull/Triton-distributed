add_bishengir_dialect_library(BiShengIRHIVMUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Utils

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  BiShengIRHIVMDialect
  MLIRMemRefDialect
  BiShengIRMemRefExtDialect
)