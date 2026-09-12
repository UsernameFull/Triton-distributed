# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Compile kernel
add_custom_target(lib_impl)
function(add_impl_share_lib NAME)
    add_library(${NAME} SHARED ${ARGN})
    target_compile_options(${NAME} PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} ${MPI_CXX_COMPILE_FLAGS} ${CCE_AICORE_ARCH})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/examples/dynamic_tiling
        ${PROJECT_SOURCE_DIR}/examples/dynamic_tiling/include
        ${PROJECT_SOURCE_DIR}/examples/templates/include
        ${PROJECT_SOURCE_DIR}/examples/utils
        ${PROJECT_SOURCE_DIR}/src
    )
    target_link_options(${NAME} PRIVATE --cce-fatobj-link)
    target_link_libraries(${NAME} PRIVATE shmem)
    add_dependencies(lib_impl ${NAME})
endfunction()

add_subdirectory(impl)

set(TILING_SOURCES
    tiling/coc_tiling_lut.cpp
    tiling/reducescatter_tiling.cpp
    tiling/allreduce_tiling.cpp
    tiling/allgather_tiling.cpp
)
add_library(tiling_lib SHARED ${TILING_SOURCES})
target_include_directories(tiling_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/examples/utils
)
target_compile_options(tiling_lib PRIVATE -O3)

add_executable(dynamic_tiling main.cpp)

target_include_directories(dynamic_tiling PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
    ${PROJECT_SOURCE_DIR}/examples/dynamic_tiling
    ${PROJECT_SOURCE_DIR}/examples/dynamic_tiling/include
    ${PROJECT_SOURCE_DIR}/examples/templates/include
    ${PROJECT_SOURCE_DIR}/examples/utils
)
target_link_options(dynamic_tiling PRIVATE --cce-fatobj-link)
target_link_libraries(dynamic_tiling PRIVATE tiling_lib shmem ${SHARE_LIB_LINK})
target_compile_options(dynamic_tiling PRIVATE -O3 ${MPI_CXX_COMPILE_FLAGS})

add_dependencies(dynamic_tiling lib_impl tiling_lib)
