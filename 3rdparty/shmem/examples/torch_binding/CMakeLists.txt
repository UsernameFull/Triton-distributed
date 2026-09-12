# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(NOT DEFINED Python3_EXECUTABLE)
    message(WARNING "Python3_EXECUTABLE is not defined, using default python3")
else()
    message("Using python: ${Python3_EXECUTABLE}")
endif()

find_package(Python3 COMPONENTS Interpreter Development)
list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}")
find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${Torch_DIR}")
message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")

if(NOT DEFINED ACLSHMEM_ALL_KERNEL_TARGETS OR ACLSHMEM_ALL_KERNEL_TARGETS STREQUAL "")
    message(FATAL_ERROR "No kernel targets passed from parent! Please check collective example build.")
else()
    foreach(TARGET IN LISTS ACLSHMEM_ALL_KERNEL_TARGETS)
        message(STATUS "  - ${TARGET}")
    endforeach()
endif()

add_library(aclshmem_torch SHARED
    src/torch_bindings.cpp
)

set_target_properties(aclshmem_torch PROPERTIES
    PREFIX ""
)

target_link_directories(aclshmem_torch PRIVATE
    ${CMAKE_BINARY_DIR}
    ${Python3_SITELIB}/torch_npu/lib
)

target_link_libraries(aclshmem_torch PRIVATE
    ${ACLSHMEM_ALL_KERNEL_TARGETS}
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    torch_npu shmem tiling_api
)

include_directories(
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_SITELIB}/torch_npu/include
    ${ACLSHMEM_SHARED_LIB_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../examples
    ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/utils
)

install(TARGETS aclshmem_torch DESTINATION lib COMPONENT aclshmem_torch)