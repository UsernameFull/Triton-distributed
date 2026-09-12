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
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
UTILS_PATH=${PROJECT_ROOT}/examples/utils
CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
PE_SIZE=${#DEVICE_ID_LIST[@]}
if [ $PE_SIZE -gt 8 ]; then
    echo "PE size is illegal"
    exit 1
fi

cd ${PROJECT_ROOT}/examples/allgather_matmul_padding/

DATA_DIR=`realpath ./out`
echo "DATA_DIR: $DATA_DIR"
EXEC_BIN=${PROJECT_ROOT}/build/bin/allgather_matmul_padding

source ${PROJECT_ROOT}/install/set_env.sh

tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N; do
    echo "Processing test case: M=${M}, K=${K}, N=${N}"

    #Generate golden data
    rm -rf ./out/*.bin
    python3 ${UTILS_PATH}/gen_data.py 1 1 ${PE_SIZE} ${M} ${N} ${K} 0 0 ${DATA_DIR}

    # Set necessary parameters
    IPPORT="tcp://127.0.0.1:8899"

    # Start Process
    export SHMEM_UID_SESSION_ID=127.0.0.1:8899
    for (( idx = 0; idx < ${PE_SIZE}; idx = idx + 1 )); do
        ${EXEC_BIN} "$PE_SIZE" "$idx" "$IPPORT" "$M" "$N" "$K" "${DATA_DIR}" "$1" &
    done

    # Wait until all process exit
    wait

    # Verify output
    python3 ${UTILS_PATH}/verify_result.py ${DATA_DIR}/aclshmem_output.bin ${DATA_DIR}/golden.bin 1 ${M} ${N} ${K} ${DATA_DIR}/torch_output.bin
    if [[ $? -ne 0 ]]; then
        exit 1
    fi
done

cd ${CURRENT_DIR}