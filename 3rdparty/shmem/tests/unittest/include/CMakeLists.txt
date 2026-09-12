# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
file(GLOB_RECURSE TEST_INCLUDE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*_kernel.cpp")

set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)

add_library(aclshmem_unittest_include OBJECT ${TEST_INCLUDE_FILES} $<TARGET_OBJECTS:aclshmem_device>)
target_link_options(aclshmem_unittest_include PRIVATE --cce-fatobj-link)
target_compile_options(aclshmem_unittest_include PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} ${CCE_AICORE_ARCH})
target_include_directories(aclshmem_unittest_include PRIVATE
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_SOURCE_DIR}/src/host
        ${PROJECT_SOURCE_DIR}/src/device
        ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/tests/unittest/include
        ${PROJECT_SOURCE_DIR}/tests/unittest/include/unittest/utils/
        ${PROJECT_SOURCE_DIR}/tests/unittest/
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/common
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/security
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/under_api
        ${PROJECT_SOURCE_DIR}/src/host/utils
        ${PROJECT_SOURCE_DIR}/src/host/mem/heap
)
