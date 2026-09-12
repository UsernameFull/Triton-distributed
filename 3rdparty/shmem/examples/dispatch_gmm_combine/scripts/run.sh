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
#bash scripts/build.sh
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
EXAMPLE_DIR=${SCRIPT_DIR}/../
UTILS_PATH=${PROJECT_ROOT}/examples/utils

# Default Args
PE_SIZE="2"
IPPORT="tcp://127.0.0.1:18878"
FIRST_NPU="0"

# Args Parse
while [[ $# -gt 0 ]]; do
    case "$1" in
        -pes)
            if [ -n "$2" ]; then
                PE_SIZE="$2"
                shift 2
            else
                echo "Error: -pes requires a value."
                exit 1
            fi
            ;;
        -fnpu)
            if [ -n "$2" ]; then
                FIRST_NPU="$2"
                shift 2
            else
                echo "Error: -fnpu requires a value."
                exit 1
            fi
            ;;
        -ipport)
            if [ -n "$2" ]; then
                IPPORT="$2"
                shift 2
            else
                echo "Error: -ipport requires a value."
                exit 1
            fi
            ;;
        -M)
            if [ -n "$2" ]; then
                M="$2"
                shift 2
            else
                echo "Error: -M requires a value."
                exit 1
            fi
            ;;
        -K)
            if [ -n "$2" ]; then
                K="$2"
                shift 2
            else
                echo "Error: -K requires a value."
                exit 1
            fi
            ;;
        -N)
            if [ -n "$2" ]; then
                N="$2"
                shift 2
            else
                echo "Error: -N requires a value."
                exit 1
            fi
            ;;
        -expertPerPe)
            if [ -n "$2" ]; then
                expertPerPe="$2"
                shift 2
            else
                echo "Error: -expertPerPe requires a value."
                exit 1
            fi
            ;;
        -dataType)
            if [ -n "$2" ]; then
                dataType="$2"
                shift 2
            else
                echo "Error: -dataType requires a value."
                exit 1
            fi
            ;;
        -weightNz)
            if [ -n "$2" ]; then
                weightNz="$2"
                shift 2
            else
                echo "Error: -weightNz requires a value."
                exit 1
            fi
            ;;
        -transB)
            if [ -n "$2" ]; then
                transB="$2"
                shift 2
            else
                echo "Error: -transB requires a value."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done

echo "DATA_DIR: $DATA_DIR"
EXEC_BIN=${PROJECT_ROOT}/build/bin/dispatch_gmm_combine

cd ${PROJECT_ROOT}/examples/dispatch_gmm_combine/

if [ -d "out" ]; then
    echo "文件夹已存在：out"
else
    echo "文件夹不存在，正在创建：out"
    mkdir -p out
fi

rm -rf out/*

python3 utils/gen_data.py
if [[ $? -ne 0 ]]; then
    echo "gen data failed."
    exit 1
fi

python3 utils/gen_data_by_torch_npu.py
if [[ $? -ne 0 ]]; then
    echo "gen data failed."
    exit 1
fi

echo "Test Case, M: ${M}, K: ${K}, N: ${N}, expertPerPe: ${expertPerPe}"
export SHMEM_UID_SESSION_ID=127.0.0.1:8899
export LD_LIBRARY_PATH=${PROJECT_ROOT}/install/shmem/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
for (( idx =0; idx < ${PE_SIZE}; idx = idx + 1 )); do
    export INPUT_PATH=${EXAMPLE_DIR}/utils/test_data_golden_cpu/
    ${EXEC_BIN} "$PE_SIZE" "$idx" "$IPPORT" "$FIRST_NPU" "$M" "$K" "$N" "$expertPerPe" "$dataType" "$weightNz" "$transB" &
done

# Wait until all process exit
wait

for (( idx =0; idx < ${PE_SIZE}; idx = idx + 1 )); do
    python3 ${UTILS_PATH}/verify_result.py ${CURRENT_DIR}/out/output_${idx}.bin \
    ${EXAMPLE_DIR}/utils/test_data_golden_cpu/unpermuted_token_${idx}.bin \
    1 ${M} ${N} ${K} \
    ${EXAMPLE_DIR}/utils/test_data_torch_npu/unpermuted_token_${idx}.bin

    if [[ $? -ne 0 ]]; then
        exit 1
    fi
done

cd ${CURRENT_DIR}