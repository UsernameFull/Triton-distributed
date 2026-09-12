#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $(dirname "$SCRIPT_DIR"))

cd ${SCRIPT_DIR}

PE_SIZE="2"
IPPORT="tcp://127.0.0.1:8766"
GNPU_NUM="8"
FIRST_NPU="0"
FIRST_PE="0"
TEST_TYPE="int"

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
                 if [[ "$GNPU_NUM" -gt "$PE_SIZE" ]]; then
                    GNPU_NUM="$PE_SIZE"
                    echo "Because GNPU_NUM is greater than PE_SIZE, GNPU_NUM is assigned the value of PE_SIZE=${PE_SIZE}."
                fi
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
        -gnpus)
            if [ -n "$2" ]; then
                GNPU_NUM="$2"
                shift 2
            else
                echo "Error: -gnpus requires a value."
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
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done

# Golden generate
rm -rf ./golden
mkdir -p golden
python3 ./utils/data_gen.py $PE_SIZE $TEST_TYPE

# Kernel test
rm -rf ./output
export SHMEM_UID_SESSION_ID=127.0.0.1:8899
# SHMEM_INSTANCE_PORT_RANGE specify the pe0 listen port range in multi-instance scene default mode
export SHMEM_INSTANCE_PORT_RANGE=1024:2047

export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
pids=()
for (( idx =0; idx < ${GNPU_NUM}; idx = idx + 1 )); do
    msprof --application="${PROJECT_ROOT}/build/bin/multi_instance $PE_SIZE $idx $IPPORT $FIRST_NPU" --output=${PROJECT_ROOT}/examples/multi_instance/output/ &
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