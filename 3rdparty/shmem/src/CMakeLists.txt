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

if (BUILD_PYTHON STREQUAL "ON")
    add_subdirectory(host/python_wrapper)
endif ()

set(ACLSHMEM_MPI_SUPPORT OFF)

if(ACLSHMEM_MPI_SUPPORT)
    find_package(MPI REQUIRED)
else()
    find_package(MPI)
    if(MPI_FOUND)
        set(ACLSHMEM_MPI_SUPPORT ON)
    endif()
endif()

file(GLOB_RECURSE ACLSHMEM_KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/device/*.cpp)
file(GLOB_RECURSE _TEMP_KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/device_simt/*.cpp)
list(APPEND ACLSHMEM_KERNEL_FILES ${_TEMP_KERNEL_FILES})
add_library(aclshmem_device OBJECT ${ACLSHMEM_KERNEL_FILES})
target_compile_options(aclshmem_device
    PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    ${CCE_AICORE_ARCH}
)
target_include_directories(aclshmem_device
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/src/device/
    ${PROJECT_SOURCE_DIR}/src/device_simt/
)

file(GLOB_RECURSE ACLSHMEM_HOST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/host/*.cpp)
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "python_wrapper")
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/bootstrap")
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/utils/log")
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/utils/under_api")
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/utils/mstx")
list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/transport/topo/rootinfo")
if(NOT ACLSHMEM_UDMA_SUPPORT)
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/transport/device_udma")
    list(FILTER ACLSHMEM_HOST_FILES EXCLUDE REGEX "host/transport/topo")
endif()

add_library(aclshmem_host OBJECT ${ACLSHMEM_HOST_FILES})
target_compile_options(aclshmem_host
    PRIVATE
    ${CMAKE_CPP_COMPILE_OPTIONS}
)
target_include_directories(aclshmem_host
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/host
    ${PROJECT_SOURCE_DIR}/src/device
    ${PROJECT_SOURCE_DIR}/src/device_simt
    ${PROJECT_SOURCE_DIR}/src/host/utils
    ${PROJECT_SOURCE_DIR}/src/host/utils/under_api
    ${PROJECT_SOURCE_DIR}/src/host/entity
    ${PROJECT_SOURCE_DIR}/src/host/mem/heap
    ${PROJECT_SOURCE_DIR}/src/host/transport
    ${PROJECT_SOURCE_DIR}/src/host/transport/device_rdma
    ${PROJECT_SOURCE_DIR}/src/host/transport/device_sdma
    ${PROJECT_SOURCE_DIR}/src/host/transport/device_udma
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
    ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
    ${PROJECT_SOURCE_DIR}/src/host/mem/heap/driver
    ${PROJECT_SOURCE_DIR}/src/host/mem/heap/driver/userspace
)
if(ACLSHMEM_UDMA_SUPPORT)
    target_include_directories(aclshmem_host PUBLIC ${PROJECT_SOURCE_DIR}/3rdparty/json/single_include)
endif()

file(GLOB_RECURSE ACLSHMEM_UTILS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/host/utils/*.cpp)
if(USE_MSSANITIZER)
    list(FILTER ACLSHMEM_UTILS_FILES EXCLUDE REGEX "host/utils/mstx/mstx_mem_register_empty.cpp")
    message(STATUS "USE_MSSANITIZER=ON")
else()
    list(FILTER ACLSHMEM_UTILS_FILES EXCLUDE REGEX "host/utils/mstx/mstx_mem_register.cpp")
    message(STATUS "USE_MSSANITIZER=OFF")
endif()
add_library(shmem_utils SHARED ${ACLSHMEM_UTILS_FILES})
target_compile_options(shmem_utils PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(shmem_utils
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/host
    ${PROJECT_SOURCE_DIR}/src/host/utils
)
if(ACLSHMEM_UDMA_SUPPORT)
    target_include_directories(shmem_utils PUBLIC ${PROJECT_SOURCE_DIR}/3rdparty/json/single_include)
endif()
install(TARGETS shmem_utils
    LIBRARY DESTINATION lib
)

add_library(shmem SHARED $<TARGET_OBJECTS:aclshmem_device> $<TARGET_OBJECTS:aclshmem_host>)
target_link_options(shmem PRIVATE --cce-fatobj-link)
if(ACLSHMEM_UDMA_SUPPORT)
    target_link_libraries(shmem PRIVATE shmem_utils shmem_rootinfo)
else()
    target_link_libraries(shmem PRIVATE shmem_utils)
endif()

# MPI
if(ACLSHMEM_MPI_SUPPORT)
    separate_arguments(ACLSHMEM_CXX_LINK_FLAGS NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
    target_link_options(shmem INTERFACE ${ACLSHMEM_CXX_LINK_FLAGS})
    target_compile_definitions(shmem INTERFACE ${MPI_CXX_COMPILE_DEFINITIONS})
    target_compile_options(shmem INTERFACE ${MPI_CXX_COMPILE_OPTIONS})

    add_library(aclshmem_bootstrap_mpi SHARED)
    target_sources(aclshmem_bootstrap_mpi PRIVATE host/bootstrap/shmemi_bootstrap_mpi.cpp)
    target_link_libraries(aclshmem_bootstrap_mpi PRIVATE MPI::MPI_CXX shmem_utils)
    target_include_directories(aclshmem_bootstrap_mpi
                                PRIVATE
                                ${PROJECT_SOURCE_DIR}/include
                                ${PROJECT_SOURCE_DIR}/src/host
    )
    set_target_properties(aclshmem_bootstrap_mpi PROPERTIES PREFIX "")
    install(TARGETS aclshmem_bootstrap_mpi
        LIBRARY DESTINATION lib
    )

endif()

# simt
if(ACLSHMEM_SIMT_SUPPORT)
    target_compile_definitions(shmem INTERFACE USE_SIMT)
endif()

# UID
add_library(aclshmem_bootstrap_uid SHARED)

target_sources(aclshmem_bootstrap_uid
    PRIVATE
    host/bootstrap/socket/uid_socket.cpp
    host/bootstrap/shmemi_bootstrap_uid.cpp
)
target_include_directories(aclshmem_bootstrap_uid
                        PRIVATE
                        ${PROJECT_SOURCE_DIR}/include
                        ${PROJECT_SOURCE_DIR}/src/host
)
set_target_properties(aclshmem_bootstrap_uid PROPERTIES PREFIX "")
target_link_libraries(aclshmem_bootstrap_uid PRIVATE shmem_utils)
install(TARGETS aclshmem_bootstrap_uid
    LIBRARY DESTINATION lib
)

# CONFIG_STORE
add_library(aclshmem_bootstrap_config_store SHARED)

file(GLOB_RECURSE CONFIG_STORE_FILES ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/*.cpp)

target_sources(aclshmem_bootstrap_config_store PRIVATE 
        ${CONFIG_STORE_FILES}
        host/bootstrap/shmemi_bootstrap_config_store.cpp
)

target_include_directories(aclshmem_bootstrap_config_store PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/host
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/include
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/common
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/security
        ${PROJECT_SOURCE_DIR}/src/host/bootstrap/config_store/acc_links/csrc/under_api/openssl
)

set_target_properties(aclshmem_bootstrap_config_store PROPERTIES PREFIX "")
target_link_libraries(aclshmem_bootstrap_config_store PRIVATE shmem_utils)
install(TARGETS aclshmem_bootstrap_config_store
        LIBRARY DESTINATION lib)


if(ACLSHMEM_UDMA_SUPPORT)
add_library(shmem_rootinfo SHARED)
file(GLOB_RECURSE ROOT_INFO_FILES ${PROJECT_SOURCE_DIR}/src/host/transport/topo/rootinfo/*.cpp ${PROJECT_SOURCE_DIR}/src/host/utils/under_api/dl_dcmi_api.cpp)

target_sources(shmem_rootinfo PRIVATE 
        ${ROOT_INFO_FILES}
)
target_compile_options(shmem_rootinfo PRIVATE
     -Werror
     -Wfloat-equal
     -Wall
     -fno-common
     -fno-strict-aliasing
     -pipe
     -O3
     -fstack-protector-all
     $<$<CONFIG:Debug>:-g>
)
 
target_link_options(shmem_rootinfo PRIVATE
     -Wl,-z,relro
     -Wl,-z,now
     -Wl,-z,noexecstack
     -s
)

target_include_directories(shmem_rootinfo PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${ASCEND_HOME_PATH}/include/
    ${ASCEND_HOME_PATH}/include/driver/
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${PROJECT_SOURCE_DIR}/src/host
    ${PROJECT_SOURCE_DIR}/src/host/utils/under_api
    ${PROJECT_SOURCE_DIR}/src/host/transport/topo
    ${PROJECT_SOURCE_DIR}/src/host/transport/topo/rootinfo
)

target_link_directories(shmem_rootinfo PRIVATE
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(shmem_rootinfo PRIVATE
    shmem_utils
    dl
    c_sec
)
install(TARGETS shmem_rootinfo
        LIBRARY DESTINATION lib)

endif()

# 安装配置
install(TARGETS shmem
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/device
    DESTINATION src
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/device_simt
    DESTINATION src
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/host_device
    DESTINATION src
)
