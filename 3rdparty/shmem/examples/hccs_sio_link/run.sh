#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $(dirname "$SCRIPT_DIR"))

EXAMPLE=hccs_sio_link

cd ${SCRIPT_DIR}

PE_SIZE="2"
IPPORT="tcp://127.0.0.1:8766"
FIRST_NPU="0"
FIRST_PE="0"
TEST_TYPE="int"
LINK_MODE="all"
DATA_SIZE_KB="4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -ipport)
            if [ -n "$2" ]; then
                IPPORT="$2"
                shift 2
            else
                echo "Error: -ipport requires a value."
                exit 1
            fi
            ;;
        -pes)
            if [ -n "$2" ]; then
                PE_SIZE="$2"
                shift 2
            else
                echo "Error: -pes requires a value."
                exit 1
            fi
            ;;
        -fpe)
            if [ -n "$2" ]; then
                FIRST_PE="$2"
                shift 2
            else
                echo "Error: -fpe requires a value."
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
        -type)
            if [ -n "$2" ]; then
                TEST_TYPE="$2"
                shift 2
            else
                echo "Error: -type requires a value."
                exit 1
            fi
            ;;
        -mode)
            if [ -n "$2" ]; then
                if [[ "$2" != "sio" && "$2" != "hccs" && "$2" != "all" && "$2" != "mixed" ]]; then
                    echo "Error: -mode only supports 'sio', 'hccs', 'all' or 'mixed'."
                    exit 1
                fi
                LINK_MODE="$2"
                shift 2
            else
                echo "Error: -mode requires a value (sio/hccs/all/mixed/perf_sio/perf_hccs/perf_all)."
                exit 1
            fi
            ;;
        -size)
            if [ -n "$2" ]; then
                DATA_SIZE_KB="$2"
                shift 2
            else
                echo "Error: -size requires a value (MB)."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done

export SHMEM_UID_SESSION_ID=127.0.0.1:8899
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

echo "============================================"
echo " HCCS Link Test"
echo " PE_SIZE:    ${PE_SIZE}"
echo " LINK_MODE:  ${LINK_MODE}"
echo " DATA_SIZE:  ${DATA_SIZE_KB} KB"
echo " TEST_TYPE:  ${TEST_TYPE}"
echo "============================================"

pids=()
for (( idx =0; idx < ${PE_SIZE}; idx = idx + 1 )); do
    ${PROJECT_ROOT}/build/bin/${EXAMPLE} $PE_SIZE $idx $IPPORT $PE_SIZE $FIRST_PE $FIRST_NPU $TEST_TYPE $LINK_MODE $DATA_SIZE_KB &
    pid=$!
    pids+=("$pid")
    echo "$pid background process recorded"
done

ret=0
for pid in ${pids[@]}; do
    wait $pid
    echo "wait process $pid done"
    cur_ret=$?
    if [[ $cur_ret -ne 0 ]]; then
        ret=$cur_ret
    fi
done

cd ${CURRENT_DIR}
exit $ret
