# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


find_program(PYTHON_EXECUTABLE NAMES python3 python)

# python
message(STATUS "PYTHON_EXECUTABLE:${PYTHON_EXECUTABLE}")
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "-c"
        "from __future__ import print_function; import os; import pybind11;
print(os.path.dirname(pybind11.__file__),end='');"
        RESULT_VARIABLE _PYTHON_SUCCESS
        OUTPUT_VARIABLE PYBIND11_DIR)

if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(STATUS "PYBIND11_DIR: ${PYBIND11_DIR}")
    message(FATAL_ERROR "Pybind11 config Error.")
endif ()

# pybind11
message(STATUS "PYBIND11_DIR = ${PYBIND11_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_DIR})
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
find_package(pybind11 REQUIRED)

set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

pybind11_add_module(_pyshmem pyshmem.cpp)
set_target_properties(_pyshmem PROPERTIES CXX_STANDARD 17)
target_include_directories(_pyshmem PRIVATE
        ${PROJECT_SOURCE_DIR}/include/
)
target_link_libraries(_pyshmem PRIVATE
        shmem
        shmem_utils
        aclshmem_bootstrap_config_store
)

set(TARGET_INSTALL_DIR ${PROJECT_SOURCE_DIR}/src/python/shmem)
install(TARGETS _pyshmem
        LIBRARY DESTINATION ${TARGET_INSTALL_DIR}/
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)
