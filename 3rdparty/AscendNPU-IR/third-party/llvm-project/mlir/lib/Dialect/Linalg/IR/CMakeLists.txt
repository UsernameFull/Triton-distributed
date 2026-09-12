set(BS_FEAT_LINK_LIBS)
set(BS_FEAT_OPTIONS)
list(APPEND BS_FEAT_OPTIONS PARTIAL_SOURCES_INTENDED)
if(BSPUB_DAVINCI_BISHENGIR)
  add_mlir_dialect_library(BiShengIRLinalgDialectExt
    LinalgExtensions.cpp

    PARTIAL_SOURCES_INTENDED

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

    DEPENDS
    MLIRLinalgInterfacesIncGen
    MLIRLinalgOpsAttributesIncGen
    MLIRLinalgOpsEnumsIncGen
    MLIRLinalgOpsIncGen
    MLIRLinalgStructuredOpsIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
  )
  list(APPEND BS_FEAT_LINK_LIBS BiShengIRLinalgDialectExt)
endif()

add_mlir_dialect_library(MLIRLinalgDialect
  LinalgInterfaces.cpp
  LinalgOps.cpp
  LinalgDialect.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  ${BS_FEAT_OPTIONS}

  DEPENDS
  MLIRLinalgInterfacesIncGen
  MLIRLinalgOpsAttributesIncGen
  MLIRLinalgOpsEnumsIncGen
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRArithUtils
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  MLIRFunctionInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRParser
  MLIRShardingInterface
  MLIRSideEffectInterfaces
  MLIRSparseTensorDialect
  MLIRSCFDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
  ${BS_FEAT_LINK_LIBS}
)
