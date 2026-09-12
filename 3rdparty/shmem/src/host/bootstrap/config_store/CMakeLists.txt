# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CONFIG_STORE_SRC_BASE ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store)
include_directories(
        ${CONFIG_STORE_SRC_BASE}/acc_links/include
        ${CONFIG_STORE_SRC_BASE}/acc_links/csrc/
        ${CONFIG_STORE_SRC_BASE}/acc_links/csrc/common
        ${CONFIG_STORE_SRC_BASE}/acc_links/csrc/security
        ${CONFIG_STORE_SRC_BASE}/acc_links/csrc/under_api/openssl
        ${CONFIG_STORE_SRC_BASE}/
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/host
        ${PROJECT_SOURCE_DIR}/src/host/utils
)

file(GLOB_RECURSE CONFIG_STORE_SRCS *.cpp)
add_library(config_store_object OBJECT ${CONFIG_STORE_SRCS})
add_library(config_store_static STATIC $<TARGET_OBJECTS:config_store_object>)
set_target_properties(config_store_static PROPERTIES OUTPUT_NAME "config_store")
set_target_properties(config_store_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
target_link_libraries(config_store_static
        -Wl,--start-group
        pthread dl rt
        -Wl,--end-group
        )

add_library(config_store_shared SHARED $<TARGET_OBJECTS:config_store_object>)
set_target_properties(config_store_shared PROPERTIES OUTPUT_NAME "config_store")
set_target_properties(config_store_shared PROPERTIES CLEAN_DIRECT_OUTPUT 1)
target_link_libraries(config_store_shared
        -Wl,--start-group
        pthread dl rt shmem_utils
        -Wl,--end-group
        )

file(GLOB HEADER_FILES ../include/*.h)
install(FILES ${HEADER_FILES} DESTINATION ${TARGET_INSTALL_DIR}/acc_links/include)
install(TARGETS config_store_shared
        LIBRARY DESTINATION lib
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        )