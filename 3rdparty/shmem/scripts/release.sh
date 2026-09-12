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

function fn_make_run_package()
{
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        ARCH="x86_64"
        echo "It is system of x86_64."
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        ARCH="aarch64"
        echo "It is system of aarch64."
    else
        echo "It is not system of x86_64 or aarch64."
        exit 1
    fi
    if [ -d "$OUTPUT_DIR/$ARCH" ]; then
        echo "$OUTPUT_DIR/$ARCH already exists."
        rm -rf "$OUTPUT_DIR/$ARCH"
        echo "$OUTPUT_DIR/$ARCH is deleted."
    fi
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    commit_id=$(git rev-parse HEAD)
    touch $OUTPUT_DIR/version.info
    cat>$OUTPUT_DIR/version.info<<EOF
        SHMEM Version :  ${VERSION}
        Platform : ${ARCH}
        branch : ${branch}
        commit id : ${commit_id}
EOF
    mkdir -p $RELEASE_DIR/$ARCH
    mkdir -p $OUTPUT_DIR/scripts
    cp $PROJECT_ROOT/scripts/install.sh $OUTPUT_DIR
    cp $PROJECT_ROOT/scripts/uninstall.sh $OUTPUT_DIR/scripts
    sed -i "s/SHMEMPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/scripts/uninstall.sh

    chmod +x $OUTPUT_DIR/*
    makeself_dir=${ASCEND_HOME_PATH}/toolkit/tools/op_project_templates/ascendc/customize/cmake/util/makeself/
    ${makeself_dir}/makeself.sh --header ${makeself_dir}/makeself-header.sh \
        --help-header $PROJECT_ROOT/scripts/help.info --gzip --complevel 4 --nomd5 --sha256 --chown \
        ${OUTPUT_DIR} $RELEASE_DIR/$ARCH/SHMEM_${VERSION}_linux-${ARCH}.run "SHMEM-api" ./install.sh
    
    rm -rf $OUTPUT_DIR/*
    mv $RELEASE_DIR/$ARCH $OUTPUT_DIR
    echo "SHMEM_${VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR"
}

function fn_main()
{
    fn_make_run_package
}
set -e
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
VERSION="1.0.0"
OUTPUT_DIR=$PROJECT_ROOT/install
THIRD_PARTY_DIR=$PROJECT_ROOT/3rdparty
RELEASE_DIR=$PROJECT_ROOT/ci/release

cann_default_path="/usr/local/Ascend/ascend-toolkit"
cd ${PROJECT_ROOT}
set +e
if [ -d "$cann_default_path" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    echo "CANN is not installed in $cann_default_path"
    exit 1
fi

set -e
fn_main "$@"