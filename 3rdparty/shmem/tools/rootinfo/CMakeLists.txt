add_executable(root_info_generate root_info_generate.cpp)

target_compile_options(root_info_generate PRIVATE
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

target_link_options(root_info_generate PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    -s
)

target_include_directories(root_info_generate PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/host/transport/topo/rootinfo
    ${ASCEND_HOME_PATH}/include/
    ${ASCEND_HOME_PATH}/include/driver/
    ${ASCEND_HOME_PATH}/pkg_inc/
)

target_link_directories(root_info_generate PRIVATE
    ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(root_info_generate PRIVATE
    shmem_rootinfo
    dl
    c_sec
)

install(TARGETS root_info_generate
    RUNTIME DESTINATION bin)