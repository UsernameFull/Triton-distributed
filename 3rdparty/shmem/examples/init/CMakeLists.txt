# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.18)
project(SHMEM)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成位置无关代码
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 设置安装路径
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install/shmem)

# 获取CANN相关环境变量
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(CMAKE_COMPILER bisheng)
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(
    -O2 -std=c++17
    -Wno-ignored-attributes
    # avoid ascendc interference
    -DL2_CACHE_HINT
    -DTILING_KEY_VAR
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${ASCEND_HOME_PATH}/compiler/tikcpp
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
)

link_directories(
    ${ASCEND_HOME_PATH}/lib64
)

link_libraries(runtime stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread)

include_directories(
    ${ASCEND_HOME_PATH}/lib64
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../install/shmem/lib
)

if(NOT DEFINED RUN_MODE)
    message(FATAL_ERROR "RUN_MODE is not set! Please specify RUN_MODE (1/2/3) when running cmake.")
endif()

if(RUN_MODE STREQUAL "1")
    set(EXECUTABLE_NAME "init_examples")
    set(COMPILE_DEF "RUN_WITH_DEFAULT")
    set(NEED_MPI "OFF")
elseif(RUN_MODE STREQUAL "2")
    find_package(MPI REQUIRED)
    include_directories(
        ${MPI_INCLUDE_PATH}
    )
    set(EXECUTABLE_NAME "init_examples")
    set(COMPILE_DEF "RUN_WITH_MPI")
    set(NEED_MPI "ON")
elseif(RUN_MODE STREQUAL "3")
    find_package(MPI REQUIRED)
    include_directories(
        ${MPI_INCLUDE_PATH}
    )
    set(EXECUTABLE_NAME "init_examples")
    set(COMPILE_DEF "RUN_WITH_UNIQUEID")
    set(NEED_MPI "ON")
elseif(RUN_MODE STREQUAL "4")
    find_package(MPI REQUIRED)
    include_directories(
        ${MPI_INCLUDE_PATH}
    )
    set(EXECUTABLE_NAME "init_examples")
    set(COMPILE_DEF "RUN_WITH_UNIQUEID_MULTI_INSTANCE")
    set(NEED_MPI "ON")
elseif(RUN_MODE STREQUAL "5")
    find_package(MPI REQUIRED)
    include_directories(
        ${MPI_INCLUDE_PATH}
    )
    set(EXECUTABLE_NAME "init_examples")
    set(COMPILE_DEF "RUN_UNIQUEID_WITH_DEFAULT")
    set(NEED_MPI "ON")
else()
    message(FATAL_ERROR "Invalid RUN_MODE! Only 1/2/3/4/5 are allowed.")
endif()

add_executable(init_examples main.cpp)

target_compile_definitions(${EXECUTABLE_NAME} PRIVATE ${COMPILE_DEF})

target_include_directories(init_examples PRIVATE
    ${ASCEND_HOME_PATH}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../install/shmem/include
)
if(NEED_MPI)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE MPI::MPI_CXX shmem)
else()
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE shmem)
endif()
