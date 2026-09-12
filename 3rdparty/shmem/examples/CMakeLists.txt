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

include_directories(
    ${PROJECT_SOURCE_DIR}/src/device/
    ${PROJECT_SOURCE_DIR}/src/device_simt/
)

# catlass 适配宏
if(SOC_TYPE STREQUAL "Ascend950")
    add_compile_definitions(CATLASS_ARCH=3510)
else()
    add_compile_definitions(CATLASS_ARCH=2201)
endif()

set(ACLSHMEM_ALL_KERNEL_TARGETS "" CACHE INTERNAL "All collective kernel shared library targets")

set(PYEXPAND_SUPPORTED_TORCH
    kv_shuffle
    allgather
    allgather_matmul
)

if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
    add_compile_definitions(ENABLE_ASCENDC_DUMP)
    add_compile_definitions(ASCENDC_DUMP=1)
    add_compile_definitions(ASCENDC_DEBUG)
    link_libraries(ascend_dump)
endif()

function(aclshmem_add_fusion_example NAME)
    if(USE_EXAMPLES)
        add_executable(${NAME} ${ARGN})
        if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
            target_compile_definitions(${NAME} PRIVATE ENABLE_ASCENDC_DUMP ASCENDC_DUMP=1 ASCENDC_DEBUG)
        endif()
        
        target_compile_options(${NAME} PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})
        target_include_directories(${NAME} PRIVATE 
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
            ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
            ${PROJECT_SOURCE_DIR}/examples/${NAME}
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/examples/templates/include
            ${PROJECT_SOURCE_DIR}/examples/utils
            ${PROJECT_SOURCE_DIR}/src/device/
            ${PROJECT_SOURCE_DIR}/src/device_simt/
            ${PROJECT_SOURCE_DIR}/src/
            ${MPI_INCLUDE_PATH}
            $ENV{ASCEND_HOME_PATH}/tools/mssanitizer/include
        )
        target_link_options(${NAME} PRIVATE --cce-fatobj-link ${CCE_AICORE_ARCH} ${SANITIZER_FLAGS})

        if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
            target_link_libraries(${NAME} PRIVATE ascend_dump)
        endif()
        target_link_libraries(${NAME} PRIVATE shmem)
        target_compile_options(${NAME} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    endif()

    if(PYEXPAND_EXAMPLE AND ${NAME} IN_LIST PYEXPAND_SUPPORTED_TORCH)
        add_library(${NAME}_kernel SHARED  ${ARGN})
        if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
            target_compile_definitions(${NAME}_kernel PRIVATE ENABLE_ASCENDC_DUMP ASCENDC_DUMP=1 ASCENDC_DEBUG)
        endif()
        
        target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220 ${SANITIZER_FLAGS})
        target_include_directories(${NAME}_kernel PRIVATE 
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
            ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
            ${PROJECT_SOURCE_DIR}/examples/${NAME}
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/examples/templates/include
            ${PROJECT_SOURCE_DIR}/examples/utils
            ${PROJECT_SOURCE_DIR}/src/device/
            ${PROJECT_SOURCE_DIR}/src/device_simt/
            ${PROJECT_SOURCE_DIR}/src/
            ${MPI_INCLUDE_PATH}
        )
        target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link --cce-aicore-arch=dav-c220 ${SANITIZER_FLAGS})

        if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
            target_link_libraries(${NAME}_kernel PRIVATE ascend_dump)
        endif()
        target_link_libraries(${NAME}_kernel PRIVATE shmem unified_dlog)
        target_compile_options(${NAME}_kernel PRIVATE ${MPI_CXX_COMPILE_FLAGS})
        
        if(${NAME} STREQUAL "dispatch_gmm_combine")
            target_compile_options(${NAME}_kernel PRIVATE --cce-auto-sync)
            target_include_directories(${NAME}_kernel PRIVATE ${PROJECT_SOURCE_DIR}/examples/${NAME}/include)
        endif()

        list(APPEND ACLSHMEM_ALL_KERNEL_TARGETS ${NAME}_kernel)
        set(ACLSHMEM_ALL_KERNEL_TARGETS ${ACLSHMEM_ALL_KERNEL_TARGETS} CACHE INTERNAL "Update kernel targets")
        install(TARGETS ${NAME}_kernel
                LIBRARY DESTINATION torch_binding/kernels)
    endif()

endfunction()

function(aclshmem_add_collective_example NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} ${CCE_AICORE_ARCH_VEC} ${SANITIZER_FLAGS})
    target_include_directories(${NAME}_kernel PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/examples/${NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/examples/utils
        ${PROJECT_SOURCE_DIR}/src/device/
        ${PROJECT_SOURCE_DIR}/src/device_simt/
        ${PROJECT_SOURCE_DIR}/src/
        $ENV{ASCEND_HOME_PATH}/tools/mssanitizer/include
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link ${CCE_AICORE_ARCH_VEC} ${SANITIZER_FLAGS})
    target_link_libraries(${NAME}_kernel PRIVATE unified_dlog)
    if(USE_EXAMPLES)
        add_executable(${NAME} main.cpp)
        target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
        target_include_directories(${NAME} PRIVATE 
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
            ${PROJECT_SOURCE_DIR}/examples/${NAME}
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/examples/templates/include
            ${PROJECT_SOURCE_DIR}/examples/utils
            ${PROJECT_SOURCE_DIR}/src/device/
            ${PROJECT_SOURCE_DIR}/src/device_simt/
            ${PROJECT_SOURCE_DIR}/src/host
            ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
            ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
            ${PROJECT_SOURCE_DIR}/src/host/mem/heap
            ${MPI_INCLUDE_PATH}
        )
        target_link_libraries(${NAME} PRIVATE shmem ${NAME}_kernel)
        target_compile_options(${NAME} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    endif()

    if(PYEXPAND_EXAMPLE AND ${NAME} IN_LIST PYEXPAND_SUPPORTED_TORCH)
        list(APPEND ACLSHMEM_ALL_KERNEL_TARGETS ${NAME}_kernel)
        set(ACLSHMEM_ALL_KERNEL_TARGETS ${ACLSHMEM_ALL_KERNEL_TARGETS} CACHE INTERNAL "Update kernel targets")
        install(TARGETS ${NAME}_kernel
                LIBRARY DESTINATION torch_binding/kernels)
    endif()

endfunction()

set(ACLSHMEM_CATLASS_FUSION_EXAMPLES
    allgather_matmul
    allgather_matmul_with_gather_result
    allgather_matmul_padding
    dispatch_gmm_combine
    matmul_allreduce
    matmul_reduce_scatter
    matmul_reduce_scatter_padding
    dynamic_tiling
)

foreach(EXAMPLE
    combine
    dispatch
    kv_shuffle
    aclgraph_demo
    allgather
    allgather_matmul
    allgather_matmul_with_gather_result
    allgather_matmul_padding
    dispatch_gmm_combine
    matmul_allreduce
    matmul_reduce_scatter
    matmul_reduce_scatter_padding
    multi_instance
    dynamic_tiling
    rma_d2h_demo
    sdma
    notifywait
    cmo
    mte_perftest
    udma_demo
    udma_atomic_add
)
    if(SOC_TYPE STREQUAL "Ascend950" AND ${EXAMPLE} IN_LIST ACLSHMEM_CATLASS_FUSION_EXAMPLES)
        message(STATUS "Skip example '${EXAMPLE}' on Ascend950 (catlass not yet supported)")
        continue()
    endif()
    add_subdirectory(${EXAMPLE})
endforeach()

if(ACLSHMEM_HCCS_SIO_LINK)
    add_subdirectory(hccs_sio_link)
endif()


if(ACLSHMEM_SIMT_SUPPORT)
foreach(
    EXAMPLE
    simt_rma
    simt_rma_scalar
)
    add_subdirectory(${EXAMPLE})
endforeach()
endif()

if(ACLSHMEM_RDMA_SUPPORT)
    foreach(EXAMPLE
        rdma_perftest
        rdma_demo
        rdma_aclgraph_demo
        rdma_handlewait_test/unuse_handlewait
        rdma_handlewait_test/use_handlewait
    )
        add_subdirectory(${EXAMPLE})
endforeach()
endif()

if(PYEXPAND_EXAMPLE)
    add_subdirectory(torch_binding)
endif()
