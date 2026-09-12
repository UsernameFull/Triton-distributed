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

add_subdirectory(device)
add_subdirectory(host)
add_subdirectory(include)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

add_executable(aclshmem_unittest $<TARGET_OBJECTS:aclshmem_unittest_host>)
target_link_directories(aclshmem_unittest PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib
)
set_target_properties(aclshmem_unittest PROPERTIES
    LINK_FLAGS "-rdynamic"
)

if(IS_NEW_VERSION)
    set(UT_LIBRARIES 
        ascendc_runtime
        mmpa
        ascend_dump
        c_sec
        error_manager
        unified_dlog
        profapi
    )
else()
    set(UT_LIBRARIES "")
endif()

target_link_libraries(aclshmem_unittest PRIVATE ${UT_LIBRARIES} aclshmem_unittest_device gtest gmock gcov aclshmem_unittest_include shmem_utils)
