# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(CMAKE_LINK_DEPENDS_NO_SHARED 1)

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER bisheng CACHE FILEPATH "C compiler")
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER bisheng CACHE FILEPATH "CXX compiler")
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)

project(SHMEM LANGUAGES C CXX)
include(CheckCXXCompilerFlag)

# 配置 ccache
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "ccache found: ${CCACHE_FOUND}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_FOUND})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_FOUND})
    execute_process(COMMAND ${CCACHE_FOUND} --version OUTPUT_VARIABLE CCACHE_V)
    message(STATUS "ccache version info: ${CCACHE_V}")
else()
    message(STATUS "ccache not found. Compilation will proceed without it.")
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成位置无关代码
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_SKIP_RPATH TRUE)

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

option(USE_UNIT_TEST "USE_UNIT_TEST" OFF)
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
option(USE_MSSANITIZER "USE_MSSANITIZER" OFF)
if(USE_CXX11_ABI)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
else()
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

if(USE_MSSANITIZER)
    if(SOC_TYPE STREQUAL "Ascend950")
        set(SANITIZER_FLAGS -g)
    else()
        set(SANITIZER_FLAGS
            -g --cce-enable-sanitizer
        )
    endif()
else()
    set(SANITIZER_FLAGS)
endif()

# AscendC head files
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(PLUGIN_VERSION 0)
# 分界版本时间戳
set(TARGET_TIMESTAMP 20260327)
set(CURRENT_TIMESTAMP 0)
set(IS_NEW_VERSION FALSE)

# 通过 bisheng -v 获取编译器版本时间
execute_process(
    COMMAND bisheng -v
    OUTPUT_VARIABLE BISHENG_STDOUT
    ERROR_VARIABLE BISHENG_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

set(BISHENG_VERSION_OUTPUT "${BISHENG_STDOUT} ${BISHENG_STDERR}")

if(BISHENG_VERSION_OUTPUT)
    message(STATUS "bisheng -v output: ${BISHENG_VERSION_OUTPUT}")

    string(REGEX MATCH "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])" DATE_MATCH "${BISHENG_VERSION_OUTPUT}")
    if(DATE_MATCH)
        string(REGEX REPLACE "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])" "\\1\\2\\3" CURRENT_TIMESTAMP "${DATE_MATCH}")
        message(STATUS "Extracted date: ${DATE_MATCH}, timestamp: ${CURRENT_TIMESTAMP}")

        if(CURRENT_TIMESTAMP GREATER_EQUAL 20260327)
            set(IS_NEW_VERSION TRUE)
            message(STATUS "Detected NEW version (${CURRENT_TIMESTAMP} >= 20260327)")
        else()
            set(IS_NEW_VERSION FALSE)
            message(STATUS "Detected OLD version (${CURRENT_TIMESTAMP} < 20260327)")
        endif()
    else()
        message(WARNING "Failed to extract date from bisheng -v output, treat as OLD version")
        set(IS_NEW_VERSION FALSE)
    endif()
else()
    message(WARNING "Failed to run bisheng -v, treat as OLD version")
    set(IS_NEW_VERSION FALSE)
endif()

message(STATUS "SOC_TYPE:${SOC_TYPE}")

if(IS_NEW_VERSION)
    set(CCE_AICORE_ARCH "")
    set(CCE_AICORE_ARCH_VEC "")
    set(FFTS_COMPILE_FLAG "")

    if(SOC_TYPE STREQUAL "Ascend950")
        set(NPU_ARCH dav-3510)
    else()
        set(NPU_ARCH dav-2201)
        set(FFTS_COMPILE_FLAG --cce-auto-set-ffts=false)
        add_compile_definitions(FFTS_CONFIG=1)
    endif()
    set(CMAKE_CCE_COMPILE_OPTIONS
        -xasc
        -Xhost-start -ftrapv -Xhost-end
        --npu-arch=${NPU_ARCH}
        --cce-auto-infer-kernel-type=false
        ${FFTS_COMPILE_FLAG}
    )
else()
    if(SOC_TYPE STREQUAL "Ascend950")
        set(CCE_AICORE_ARCH --cce-aicore-arch=dav-c310)
        set(CCE_AICORE_ARCH_VEC --cce-aicore-arch=dav-c310-vec)
    else()
        set(CCE_AICORE_ARCH --cce-aicore-arch=dav-c220)
        set(CCE_AICORE_ARCH_VEC --cce-aicore-arch=dav-c220-vec)
        add_compile_definitions(FFTS_CONFIG=1)
    endif()
    set(NPU_ARCH "")
    set(CMAKE_CCE_COMPILE_OPTIONS
        -xcce
        -Xhost-start -ftrapv -Xhost-end
        "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
        "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
        "SHELL:-mllvm -cce-aicore-record-overflow=true"
        "SHELL:-mllvm -cce-aicore-addr-transform"
        "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
    )
endif()
message(STATUS "CMAKE_CCE_COMPILE_OPTIONS: ${CMAKE_CCE_COMPILE_OPTIONS}")

option(USE_EXAMPLES "USE_EXAMPLES" OFF)
option(ENABLE_ASCENDC_DUMP "ENABLE_ASCENDC_DUMP" OFF)
message(STATUS "USE_UNIT_TEST:${USE_UNIT_TEST}")
message(STATUS "USE_EXAMPLES:${USE_EXAMPLES}")
message(STATUS "ENABLE_ASCENDC_DUMP:${ENABLE_ASCENDC_DUMP}")
option(ACLSHMEM_RDMA_SUPPORT "ACLSHMEM_RDMA_SUPPORT" OFF)
message(STATUS "ACLSHMEM_RDMA_SUPPORT:${ACLSHMEM_RDMA_SUPPORT}")

if(DEFINED ACLSHMEM_RDMA_BACKEND AND NOT ACLSHMEM_RDMA_SUPPORT)
    message(FATAL_ERROR "ACLSHMEM_RDMA_BACKEND requires ACLSHMEM_RDMA_SUPPORT to be ON. Please add -DACLSHMEM_RDMA_SUPPORT=ON to enable RDMA support.")
endif()

if(ACLSHMEM_RDMA_SUPPORT)
    if(SOC_TYPE STREQUAL "Ascend950")
        if(NOT DEFINED ACLSHMEM_RDMA_BACKEND)
            message(FATAL_ERROR "ACLSHMEM_RDMA_BACKEND is not set. Please specify -rdma_backend with value 'XSCALE' when SOC_TYPE is Ascend950.")
        endif()

        string(TOUPPER "${ACLSHMEM_RDMA_BACKEND}" RDMA_BACKEND_UPPER)
        if(RDMA_BACKEND_UPPER STREQUAL "XSCALE")
            add_compile_definitions(ACLSHMEMI_RDMA_K_BACKEND_XSCALE=1)
            message(STATUS "ACLSHMEM_RDMA_BACKEND: XSCALE")
        else()
            message(FATAL_ERROR "Invalid ACLSHMEM_RDMA_BACKEND value '${ACLSHMEM_RDMA_BACKEND}'. Must be 'XSCALE'.")
        endif()
    elseif(DEFINED ACLSHMEM_RDMA_BACKEND)
        message(FATAL_ERROR "ACLSHMEM_RDMA_BACKEND can only be specified when SOC_TYPE is Ascend950.")
    endif()
endif()

option(ACLSHMEM_HCCS_SIO_LINK "ACLSHMEM_HCCS_SIO_LINK" OFF)

option(ACLSHMEM_SIMT_SUPPORT "ACLSHMEM_SIMT_SUPPORT" OFF)
message(STATUS "ACLSHMEM_SIMT_SUPPORT:${ACLSHMEM_SIMT_SUPPORT}")

option(ACLSHMEM_UDMA_SUPPORT "ACLSHMEM_UDMA_SUPPORT" OFF)
message(STATUS "ACLSHMEM_UDMA_SUPPORT:${ACLSHMEM_UDMA_SUPPORT}")

option(PYEXPAND_EXAMPLE "PYEXPAND_EXAMPLE" OFF)
message(STATUS "PYEXPAND_EXAMPLE:${PYEXPAND_EXAMPLE}")

message(STATUS "CMAKE_BUILD_TYPE:${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG_MODE)
endif()

if(ACLSHMEM_UDMA_SUPPORT)
    add_compile_definitions(ACLSHMEM_UDMA_SUPPORT)
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Werror
    # avoid ascendc interference
    -DL2_CACHE_HINT
    -DTILING_KEY_VAR
    -fstack-protector-strong
)

check_cxx_compiler_flag("-Wno-unused-value" HAS_WNO_UNUSED_VALUE)
if(HAS_WNO_UNUSED_VALUE)
    add_compile_options(-Wno-unused-value)
endif()

check_cxx_compiler_flag("-Wno-macro-redefined" HAS_WNO_MACRO_REDEFINED)
if(HAS_WNO_MACRO_REDEFINED)
    add_compile_options(-Wno-macro-redefined)
endif()

check_cxx_compiler_flag("-Wno-ignored-attributes" HAS_WNO_IGNORED_ATTRIBUTES)
if(HAS_WNO_IGNORED_ATTRIBUTES)
    add_compile_options(-Wno-ignored-attributes)
endif()

add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
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
    ${ASCEND_HOME_PATH}/include/aclnn
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof

    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/

    # adapt to CANN eco package
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/profiling
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/interface
)

link_directories(
    ${ASCEND_HOME_PATH}/lib64
)

link_libraries(runtime stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread)

# Check if aclrtMemExportToShareableHandleV2 symbol exists in the ACL library
include(CheckCXXSymbolExists)

# Create a test file to check for the symbol
set(CMAKE_REQUIRED_LIBRARIES ascendcl)
set(CMAKE_REQUIRED_INCLUDES ${ASCEND_HOME_PATH}/include)
set(CMAKE_REQUIRED_LINK_OPTIONS -L${ASCEND_HOME_PATH}/lib64)

# Try to compile a simple program that uses the symbol
file(WRITE ${CMAKE_BINARY_DIR}/check_acl_symbol.cpp "
#include <acl/acl.h>
int main() {
    aclrtMemFabricHandle handle;
    aclrtMemExportToShareableHandleV2(nullptr, 1, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, &handle);
    aclrtMemImportFromShareableHandleV2(&handle, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, 0, nullptr);
    return 0;
}")

# Check if we can link the program with the symbol
try_compile(HAS_ACLRT_MEM_FABRIC_HANDLE
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/check_acl_symbol.cpp
    LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_REQUIRED_INCLUDES}" "-DLINK_DIRECTORIES=${ASCEND_HOME_PATH}/lib64"
    OUTPUT_VARIABLE COMPILE_OUTPUT
)

# Clean up the test file
file(REMOVE ${CMAKE_BINARY_DIR}/check_acl_symbol.cpp)

option(ENABLE_CANN_BUILD "Enable CANN Build" OFF)

if(HAS_ACLRT_MEM_FABRIC_HANDLE AND ENABLE_CANN_BUILD)
    message(STATUS "aclrtMemExportToShareableHandleV2 symbol found, enabling ACLRT_MEM_FABRIC_HANDLE support")
    add_definitions(-DHAS_ACLRT_MEM_FABRIC_HANDLE)
else()
    message(STATUS "aclrtMemExportToShareableHandleV2 symbol not found, disabling ACLRT_MEM_FABRIC_HANDLE support")
endif()

# Check if aclrtMemMapSelectedLink symbol exists in the ACL library
file(WRITE ${CMAKE_BINARY_DIR}/check_aclrt_mem_map_selected_link.cpp "
#include <acl/acl.h>
int main() {
    void *ptr = nullptr;
    aclrtMemMapSelectedLink(ptr, 0, ptr, 0);
    return 0;
}")

try_compile(HAS_ACLRT_MEM_MAP_SELECTED_LINK
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/check_aclrt_mem_map_selected_link.cpp
    LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_REQUIRED_INCLUDES}" "-DLINK_DIRECTORIES=${ASCEND_HOME_PATH}/lib64"
    OUTPUT_VARIABLE COMPILE_OUTPUT
)

file(REMOVE ${CMAKE_BINARY_DIR}/check_aclrt_mem_map_selected_link.cpp)

if(HAS_ACLRT_MEM_MAP_SELECTED_LINK)
    set(ACLSHMEM_HCCS_SIO_LINK ON)
    message(STATUS "aclrtMemMapSelectedLink symbol found, enabling ACLSHMEM_HCCS_SIO_LINK support")
else()
    set(ACLSHMEM_HCCS_SIO_LINK OFF)
    message(STATUS "aclrtMemMapSelectedLink symbol not found, disabling ACLSHMEM_HCCS_SIO_LINK support")
endif()

# 添加子目录
add_subdirectory(src)
add_subdirectory(tools)

if(USE_UNIT_TEST)
    add_subdirectory(tests/unittest)
endif()

if(USE_EXAMPLES OR PYEXPAND_EXAMPLE)
    add_subdirectory(examples)
endif()
