# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_executable(hccs_sio_link main.cpp hccs_sio_link_kernel.cpp)
if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
    target_compile_definitions(hccs_sio_link PRIVATE ENABLE_ASCENDC_DUMP ASCENDC_DUMP=1 ASCENDC_DEBUG)
endif()
target_compile_options(hccs_sio_link PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})
target_include_directories(hccs_sio_link PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
    ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
    ${PROJECT_SOURCE_DIR}/examples/hccs_sio_link
    ${PROJECT_SOURCE_DIR}/examples/templates/include
    ${PROJECT_SOURCE_DIR}/examples/utils
    ${PROJECT_SOURCE_DIR}/src/device/
    ${PROJECT_SOURCE_DIR}/src/device_simt/
    ${PROJECT_SOURCE_DIR}/src/
    ${MPI_INCLUDE_PATH}
    $ENV{ASCEND_HOME_PATH}/tools/mssanitizer/include
)
target_link_options(hccs_sio_link PRIVATE --cce-fatobj-link ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})
if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
    target_link_libraries(hccs_sio_link PRIVATE ascend_dump)
endif()
target_link_libraries(hccs_sio_link PRIVATE shmem)
target_compile_options(hccs_sio_link PRIVATE ${MPI_CXX_COMPILE_FLAGS})
