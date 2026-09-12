#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

ascend_dir=$(dirname "$_ASCEND_INSTALL_PATH")
env_script_path_old="${ascend_dir}/set_env.sh"
env_script_path_new="${ascend_dir}/ascend-toolkit/set_env.sh"

if [ -n "$_ASCEND_INSTALL_PATH" ] && [ -f "$env_script_path_old" ] && [ -x "$env_script_path_old" ] && \
   [ -f "$env_script_path_new" ] && [ -x "$env_script_path_new" ]; then
    echo "[WARNING] Both old and new set_env.sh files are detected!"
    echo "          Old path: $env_script_path_old"
    echo "          New path: $env_script_path_new"
    echo "          The new path file will be used by priority!"
fi

if [ -n "$_ASCEND_INSTALL_PATH" ] && [ -f "$env_script_path_new" ] && [ -x "$env_script_path_new" ]; then
    source "$env_script_path_new"
elif [ -n "$_ASCEND_INSTALL_PATH" ] && [ -f "$env_script_path_old" ] && [ -x "$env_script_path_old" ]; then
    source "$env_script_path_old"
else
    if [ -z "$_ASCEND_INSTALL_PATH" ]; then
        echo "[WARNING] Environment variable _ASCEND_INSTALL_PATH is not set, cannot find set_env.sh script" >&2
    else
        echo "[WARNING] Valid set_env.sh script not found!" >&2
        echo "       Check path 1: $env_script_path_old (does not exist or is not executable)" >&2
        echo "       Check path 2: $env_script_path_new (does not exist or is not executable)" >&2
    fi
fi

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
VERSION="1.0.0"
OUTPUT_DIR=$PROJECT_ROOT/install
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
THIRD_PARTY_DIR=$PROJECT_ROOT/3rdparty
mkdir -p $THIRD_PARTY_DIR
RELEASE_DIR=$PROJECT_ROOT/ci/release
UNDER_DIR=$PROJECT_ROOT/src/

BUILD_TYPE=RELEASE
PYEXPAND_TYPE=OFF
PACKAGE=OFF
USE_CXX11_ABI=ON
USE_MSSANITIZER=OFF
ENABLE_EXAMPLES=OFF
PYEXPAND_EXAMPLE=OFF
BUILD_ALL=OFF

COMPILE_OPTIONS=""

COVERAGE_TYPE=""
GEN_DOC=OFF
SOC_TYPE=""
RDMA_BACKEND=""

cann_default_path="/usr/local/Ascend/ascend-toolkit"

cd ${PROJECT_ROOT}

function print_usage()
{
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -uttests                   Build with unit tests"
    echo "  -cann                      Enable CANN build"
    echo "  -debug                     Build in debug mode"
    echo "  -examples                  Build with examples"
    echo "  -enable_rdma               Enable RDMA support. For Ascend950, must be used with -rdma_backend"
    echo "  -enable_simt               Enable SIMT support"
    echo "  -python_extension          Build Python extension"
    echo "  -python_example            Build Python example"
    echo "  -gendoc                    Generate documentation"
    echo "  -onlygendoc                Only generate documentation"
    echo "  -enable_ascendc_dump       Enable AscendC dump"
    echo "  -package                   Build package"
    echo "  -full                      Full build (all components)"
    echo "  -use_cxx11_abi1            Use CXX11 ABI=1"
    echo "  -use_cxx11_abi0            Use CXX11 ABI=0"
    echo "  -mssanitizer               Enable memory sanitizer"
    echo "  -soc_type <type>           Specify SOC type (e.g., Ascend950)"
    echo "  -rdma_backend <backend>    Specify RDMA backend (XSCALE, only for Ascend950)"
    echo "                             Requires -enable_rdma and -soc_type Ascend950"
}

function fn_build()
{
    mkdir -p build && cd build
    local enable_udma_support=OFF
    if [ "$SOC_TYPE" = "Ascend950" ]; then
        enable_udma_support=ON
    fi

    cmake $COMPILE_OPTIONS -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DUSE_CXX11_ABI=$USE_CXX11_ABI -DUSE_MSSANITIZER=$USE_MSSANITIZER -DSOC_TYPE=${SOC_TYPE} -DPYEXPAND_EXAMPLE=$PYEXPAND_EXAMPLE -DACLSHMEM_UDMA_SUPPORT=$enable_udma_support ..
    make install -j17
    cd -
}

function fn_whl_build()
{
    echo "Python extension enabled. Copying and packaging Python wheel..."
    export SOC_TYPE=${SOC_TYPE}
    if [ "$SOC_TYPE" = "Ascend950" ]; then
        export ACLSHMEM_UDMA_SUPPORT=ON
    fi

    cd "${PROJECT_ROOT}/src/python"
    rm -rf shmem.egg-info ${PROJECT_ROOT}/dist

    GIT_COMMIT=`git rev-parse HEAD` || true
    {
        echo "commit_id: ${GIT_COMMIT}"
    } > "${PROJECT_ROOT}/src/python/shmem/VERSION"

    cd "${PROJECT_ROOT}"
    python3 setup.py bdist_wheel
}

function make_package()
{
    rm -rf "${PROJECT_ROOT}/package"
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        ARCH="aarch64"
    else
        exit 1
    fi

    mkdir -p "${PROJECT_ROOT}"/package/$ARCH/
    if [ "$PYEXPAND_TYPE" = "ON" ]; then
         cp "${PROJECT_ROOT}"/dist/*.whl "${PROJECT_ROOT}"/package/$ARCH/
         whl_name=`basename ${PROJECT_ROOT}/src/python/dist/*.whl`
         echo "${whl_name} is copy to ${PROJECT_ROOT}/package"
    fi
    cp -r "${PROJECT_ROOT}"/install/$ARCH "${PROJECT_ROOT}"/package
    echo "SHMEM_${VERSION}_linux-${ARCH}.run is copy to ${PROJECT_ROOT}/package"
}

function fn_make_run_package()
{
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        echo "it is system of x86_64"
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        echo "it is system of aarch64"
        ARCH="aarch64"
    else
        echo "it is not system of x86_64 or aarch64"
        exit 1
    fi

    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    commit_id=$(git rev-parse HEAD) || true
    mkdir -p $OUTPUT_DIR
    touch $OUTPUT_DIR/version.info
    cat>$OUTPUT_DIR/version.info<<EOF
        SHMEM Version :  ${VERSION}
        Platform : ${ARCH}
        branch : ${branch}
        commit id : ${commit_id}
EOF

    mkdir -p $OUTPUT_DIR/scripts
    mkdir -p $RELEASE_DIR/$ARCH
    cp $PROJECT_ROOT/scripts/install.sh $OUTPUT_DIR
    cp $PROJECT_ROOT/scripts/set_env.sh $OUTPUT_DIR
    cp $PROJECT_ROOT/scripts/uninstall.sh $OUTPUT_DIR/scripts

    sed -i "s/SHMEMPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/scripts/uninstall.sh

    chmod +x $OUTPUT_DIR/*.sh

    makeself_dir=${ASCEND_HOME_PATH}/toolkit/tools/op_project_templates/ascendc/customize/cmake/util/makeself/
    ${makeself_dir}/makeself.sh --header ${makeself_dir}/makeself-header.sh \
        --help-header $PROJECT_ROOT/scripts/help.info --gzip --complevel 4 --nomd5 --sha256 --chown \
        ${OUTPUT_DIR} $RELEASE_DIR/$ARCH/SHMEM_${VERSION}_linux-${ARCH}.run "SHMEM-api" ./install.sh
    [ -d "$OUTPUT_DIR/$ARCH" ] && rm -rf "$OUTPUT_DIR/$ARCH"
    cp -r $RELEASE_DIR/$ARCH $OUTPUT_DIR
    echo "SHMEM_${VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR/$ARCH"
}

function fn_build_googletest()
{
    if [ -d "$THIRD_PARTY_DIR/googletest/lib" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    [[ ! -d "googletest" ]] && git clone --branch v1.14.x --depth 1 https://gitcode.com/GitHub_Trending/go/googletest.git
    cd googletest

    mkdir -p build && cd build
    if [ "$USE_CXX11_ABI" == "ON" ]
    then
        sed -i '21 a add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)' ../CMakeLists.txt
    else
        sed -i '21 a add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)' ../CMakeLists.txt
    fi

    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/googletest -DCMAKE_SKIP_RPATH=TRUE -DCMAKE_CXX_FLAGS="-fPIC"
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
    [[ -d "$THIRD_PARTY_DIR/googletest/lib64" ]] && cp -rf $THIRD_PARTY_DIR/googletest/lib64 $THIRD_PARTY_DIR/googletest/lib
    echo "Googletest is successfully installed to $THIRD_PARTY_DIR/googletest"
    cd ${PROJECT_ROOT}
}

function fn_build_nlohmann_json()
{
    if [ -f "$THIRD_PARTY_DIR/json/single_include/nlohmann/json.hpp" ]; then
        return 0
    fi

    cd $THIRD_PARTY_DIR
    rm -rf json
    git clone --branch v3.11.3 --depth 1 https://github.com/nlohmann/json.git json
    cd ${PROJECT_ROOT}
}

function fn_build_doxygen()
{
    if [ -d "$THIRD_PARTY_DIR/doxygen" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    wget --no-check-certificate https://github.com/doxygen/doxygen/releases/download/Release_1_9_6/doxygen-1.9.6.src.tar.gz
    tar -xzvf doxygen-1.9.6.src.tar.gz
    cd doxygen-1.9.6
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/doxygen
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
    rm -rf $THIRD_PARTY_DIR/doxygen-1.9.6
    cd ${PROJECT_ROOT}
}

function fn_build_sphinx()
{
    [[ "$COVERAGE_TYPE" != "" ]] && return 0
    pip install sphinx
    pip install sphinx_rtd_theme
    pip install myst_parser
    pip install breathe
    pip install linkify-it-py
}

function fn_gen_doc()
{
    cd $PROJECT_ROOT
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    local doxyfile=$PROJECT_ROOT/docs/Doxyfile
    local doxygen_output_dir=$PROJECT_ROOT/docs/$branch
    [[ -f "$doxyfile" ]] && rm -rf $doxyfile
    [[ -d "$doxygen_output_dir" ]] && rm -rf $doxygen_output_dir
    mkdir -p $doxygen_output_dir
    $THIRD_PARTY_DIR/doxygen/bin/doxygen -g $doxyfile
    sed -i "s#PROJECT_NAME           =.*#PROJECT_NAME           = \"Shmem\"#g" $doxyfile
    sed -i "s#PROJECT_NUMBER         =.*#PROJECT_NUMBER         = $branch#g" $doxyfile
    sed -i "s#OUTPUT_DIRECTORY       =.*#OUTPUT_DIRECTORY       = $doxygen_output_dir#g" $doxyfile
    sed -i "s#OUTPUT_LANGUAGE        =.*#OUTPUT_LANGUAGE        = English#g" $doxyfile
    sed -i "s#INPUT                  =.*#INPUT                  = $PROJECT_ROOT/include/host $PROJECT_ROOT/include/device $PROJECT_ROOT/include/host_device#g" $doxyfile
    sed -i "s#RECURSIVE              =.*#RECURSIVE              = YES#g" $doxyfile
    sed -i "s#USE_MDFILE_AS_MAINPAGE =.*#USE_MDFILE_AS_MAINPAGE = $PROJECT_ROOT/README.md#g" $doxyfile
    sed -i "s#HTML_EXTRA_STYLESHEET  =.*#HTML_EXTRA_STYLESHEET  = $PROJECT_ROOT/docs/doxygen/custom.css#g" $doxyfile
    sed -i "s#GENERATE_LATEX         =.*#GENERATE_LATEX         = NO#g" $doxyfile
    sed -i "s#HAVE_DOT               =.*#HAVE_DOT               = NO#g" $doxyfile
    sed -i "s#WARNINGS_AS_ERROR      =.*#WARNINGS_AS_ERROR      = NO#g" $doxyfile
    sed -i "s#EXTRACT_ALL            =.*#EXTRACT_ALL            = YES#g" $doxyfile
    sed -i "s#USE_MATHJAX            =.*#USE_MATHJAX            = YES#g" $doxyfile
    sed -i "s#WARN_NO_PARAMDOC       =.*#WARN_NO_PARAMDOC       = YES#g" $doxyfile
    sed -i "s#GENERATE_TREEVIEW      =.*#GENERATE_TREEVIEW      = YES#g" $doxyfile
    sed -i "s#WARN_AS_ERROR          =.*#WARN_AS_ERROR          = YES#g" $doxyfile
    sed -i "s#GENERATE_XML           =.*#GENERATE_XML           = YES#g" $doxyfile
    sed -i "s#EXPAND_ONLY_PREDEF     =.*#EXPAND_ONLY_PREDEF     = NO#g" $doxyfile
    sed -i "s#SKIP_FUNCTION_MACROS   =.*#SKIP_FUNCTION_MACROS   = NO#g" $doxyfile
    sed -i "s#ALLOW_DUPLICATE_MEMBERS =.*#ALLOW_DUPLICATE_MEMBERS = YES#g" $doxyfile
    sed -i "s#EXCLUDE_SYMBOLS        =.*#EXCLUDE_SYMBOLS        = shmem* addrGm#g" "$doxyfile"
    $THIRD_PARTY_DIR/doxygen/bin/doxygen $doxyfile
    [[ "$COVERAGE_TYPE" != "" ]] && return 0
    local sphinx_out_dir=$PROJECT_ROOT/docs/$branch/guide
    [[ -d "$sphinx_out_dir" ]] && rm -rf $sphinx_out_dir
    mkdir -p $sphinx_out_dir
    sphinx-build -M html $PROJECT_ROOT/docs $sphinx_out_dir
}

set -e
while [[ $# -gt 0 ]]; do
    case "$1" in
        -uttests)
            fn_build_googletest
            BUILD_TYPE=Debug
            cd $THIRD_PARTY_DIR; [[ ! -d "catlass" ]] && git clone https://gitcode.com/cann/catlass.git; cd $PROJECT_ROOT
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_UNIT_TEST=ON"
            shift
            ;;
        -cann)
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DENABLE_CANN_BUILD=ON"
            shift
            ;;
        -debug)
            BUILD_TYPE=Debug
            COMPILE_OPTIONS="${COMPILE_OPTIONS}"
            shift
            ;;
        -examples)
            cd $THIRD_PARTY_DIR; [[ ! -d "catlass" ]] && git clone https://gitcode.com/cann/catlass.git; cd $PROJECT_ROOT
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_EXAMPLES=ON -DPython3_EXECUTABLE=$(which python3)"
            ENABLE_EXAMPLES=ON
            shift
            ;;
        -enable_rdma)
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DACLSHMEM_RDMA_SUPPORT=ON"
            shift
            ;;
        -enable_simt)
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DACLSHMEM_SIMT_SUPPORT=ON"
            shift
            ;;
        -python_extension)
            PYEXPAND_TYPE=ON
            shift
            ;;
        -python_example)
            cd $THIRD_PARTY_DIR; [[ ! -d "catlass" ]] && git clone https://gitcode.com/cann/catlass.git; cd $PROJECT_ROOT
            PYEXPAND_EXAMPLE=ON
            shift
            ;;
        -gendoc)
            fn_build_doxygen
            fn_build_sphinx
            GEN_DOC=ON
            shift
            ;;
        -onlygendoc)
            fn_build_doxygen
            fn_build_sphinx
            fn_gen_doc
            exit 0
            shift
            ;;
        -enable_ascendc_dump)
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DENABLE_ASCENDC_DUMP=ON"
            shift
            ;;
        -package)
            PACKAGE=ON
            PYEXPAND_TYPE=ON
            shift
            ;;
        -full)
            BUILD_ALL=ON
            fn_build_googletest
            cd $THIRD_PARTY_DIR; [[ ! -d "catlass" ]] && git clone https://gitcode.com/cann/catlass.git; cd $PROJECT_ROOT
            shift
            ;;
        -use_cxx11_abi1)
            USE_CXX11_ABI=ON
            shift
            ;;
        -use_cxx11_abi0)
            USE_CXX11_ABI=OFF
            shift
            ;;
        -mssanitizer)
            USE_MSSANITIZER=ON
            shift
            ;;
        -soc_type)
            SOC_TYPE="$2"
            shift 2
            ;;
        -rdma_backend)
            if [ "$RDMA_BACKEND" != "" ]; then
                echo "Error: -rdma_backend can only be specified once."
                print_usage
                exit 1
            fi
            shift
            RDMA_BACKEND="$1"
            if [ "$RDMA_BACKEND" != "XSCALE" ]; then
                echo "Error: Invalid RDMA_BACKEND value '$RDMA_BACKEND'. Must be 'XSCALE'."
                print_usage
                exit 1
            fi
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DACLSHMEM_RDMA_BACKEND=${RDMA_BACKEND}"
            shift
            ;;
        *)
            echo "Error: Unknown option $1."
            print_usage
            exit 1
            ;;
    esac
done

# Validate RDMA parameter dependencies
if [ -n "$RDMA_BACKEND" ]; then
    if [ "$SOC_TYPE" != "Ascend950" ]; then
        echo "Error: -rdma_backend can only be specified when SOC_TYPE is Ascend950."
        echo "       Please use -soc_type Ascend950 when specifying -rdma_backend."
        print_usage
        exit 1
    fi
    if [ -z "$(echo "$COMPILE_OPTIONS" | grep -o '\-DACLSHMEM_RDMA_SUPPORT=ON')" ]; then
        echo "Error: -rdma_backend requires -enable_rdma to be specified."
        echo "       Please add -enable_rdma flag when using -rdma_backend."
        print_usage
        exit 1
    fi
fi

if [ "$SOC_TYPE" = "Ascend950" ]; then
    if [ -n "$(echo "$COMPILE_OPTIONS" | grep -o '\-DACLSHMEM_RDMA_SUPPORT=ON')" ]; then
        if [ -z "$RDMA_BACKEND" ]; then
            echo "Error: -rdma_backend must be specified when SOC_TYPE is Ascend950 and RDMA is enabled."
            print_usage
            exit 1
        fi
    fi
    fn_build_nlohmann_json
fi

# 清空 build
[ -d build ] && rm -rf build

if [ "$BUILD_ALL" = "ON" ]; then
    OLD_COMPILE_OPTIONS=${COMPILE_OPTIONS}
    # build whl
    fn_whl_build

    # build examples
    COMPILE_OPTIONS="${OLD_COMPILE_OPTIONS} -DUSE_EXAMPLES=ON"
    fn_build

    # build uttests
    BUILD_TYPE=Debug
    COMPILE_OPTIONS="${OLD_COMPILE_OPTIONS} -DUSE_UNIT_TEST=ON"
    fn_build

    # build package
    PYEXPAND_TYPE=ON
    fn_make_run_package
    make_package
else
    if [ "$PYEXPAND_TYPE" = "ON" ]; then
        fn_whl_build
    fi

    fn_build
    fn_make_run_package
    if [ "$PACKAGE" == "ON" ]; then
        make_package
    fi

    if [ ${GEN_DOC} == "ON" ]; then
        fn_gen_doc
    fi
fi

cd ${CURRENT_DIR}
