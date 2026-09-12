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
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

cd ${SCRIPT_DIR}

export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

EXEC_BIN=${PROJECT_ROOT}/build/bin/shmem_perftest

# 默认测试类型为put
TEST_TYPE="put"
# 默认数据类型为float
DATA_TYPE="float"
# 默认核数范围
MIN_BLOCK_SIZE="32"
MAX_BLOCK_SIZE="32"
# 默认幂数范围
MIN_EXPONENT="3"
MAX_EXPONENT="17"
# 默认循环次数
LOOP_COUNT="1000"
# 默认UB size(KB)
UB_SIZE="16"
# 默认SHMEM内存类型: hbm/dram
MEMORY_TYPE="hbm"
# 默认RANK配置
PE_SIZE="2"
IPPORT="tcp://127.0.0.1:8767"
GNPU_NUM="2"
FIRST_NPU="0"
FIRST_PE="0"
# 分析模式: none/plot/md (none表示都不生成, plot表示只生成图, md表示同时生成图和md), 默认none
ANALYSE_MODE="none"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test-type)
            if [ -n "$2" ]; then
                TEST_TYPE="$2"
                shift 2
            else
                echo "Error: -t|--test-type requires a value."
                exit 1
            fi
            ;;
        -d|--datatype)
            if [ -n "$2" ]; then
                DATA_TYPE="$2"
                shift 2
            else
                echo "Error: -d|--datatype requires a value."
                exit 1
            fi
            ;;
        --loop-count)
            if [ -n "$2" ]; then
                LOOP_COUNT="$2"
                shift 2
            else
                echo "Error: --loop-count requires a value."
                exit 1
            fi
            ;;
        --ub-size)
            if [ -n "$2" ]; then
                UB_SIZE="$2"
                shift 2
            else
                echo "Error: --ub-size requires a value."
                exit 1
            fi
            ;;
        --memory-type)
            if [ -n "$2" ]; then
                MEMORY_TYPE="$2"
                shift 2
            else
                echo "Error: --memory-type requires a value."
                exit 1
            fi
            ;;
        -b|--block-size)
            if [ -n "$2" ]; then
                MIN_BLOCK_SIZE="$2"
                MAX_BLOCK_SIZE="$2"
                shift 2
            else
                echo "Error: -b|--block-size requires a value."
                exit 1
            fi
            ;;
        --block-range)
            if [ -n "$2" ] && [ -n "$3" ]; then
                MIN_BLOCK_SIZE="$2"
                MAX_BLOCK_SIZE="$3"
                shift 3
            else
                echo "Error: --block-range requires two values."
                exit 1
            fi
            ;;
        -e|--exponent)
            if [ -n "$2" ]; then
                MIN_EXPONENT="$2"
                MAX_EXPONENT="$2"
                shift 2
            else
                echo "Error: -e|--exponent requires a value."
                exit 1
            fi
            ;;
        --exponent-range)
            if [ -n "$2" ] && [ -n "$3" ]; then
                MIN_EXPONENT="$2"
                MAX_EXPONENT="$3"
                shift 3
            else
                echo "Error: --exponent-range requires two values."
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
        -ipport)
            if [ -n "$2" ]; then
                IPPORT="$2"
                shift 2
            else
                echo "Error: -ipport requires a value."
                exit 1
            fi
            ;;
        -gnpus)
            if [ -n "$2" ]; then
                GNPU_NUM="$2"
                if [[ "$GNPU_NUM" -gt "$PE_SIZE" ]]; then
                    GNPU_NUM="$PE_SIZE"
                    echo "Because GNPU_NUM is greater than PE_SIZE, GNPU_NUM is assigned the value of PE_SIZE=${PE_SIZE}."
                fi
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
        -fpe)
            if [ -n "$2" ]; then
                FIRST_PE="$2"
                shift 2
            else
                echo "Error: -fpe requires a value."
                exit 1
            fi
            ;;
        -a|--analyse)
            if [ -n "$2" ]; then
                ANALYSE_MODE="$2"
                shift 2
            else
                echo "Error: -a|--analyse requires a value."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1."
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  -t|--test-type <type>           设置测试类型 (put|bi_put|get|bi_get|all)"
            echo "  -d|--datatype <type>            设置数据类型 (float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char|all)"
            echo "  -b|--block-size <size>          设置核数"
            echo "  --block-range <min> <max>       设置核数范围"
            echo "  -e|--exponent <exponent>        设置数据量的幂数"
            echo "  --exponent-range <min> <max>    设置数据量的幂数范围"
            echo "  --loop-count <count>            设置循环次数"
            echo "  --ub-size <size>                设置UB size(KB), 默认16"
            echo "  --memory-type <hbm|dram>        设置SHMEM内存类型, 默认hbm"
            echo "  -pes <size>                     设置PE大小"
            echo "  -ipport <ip:port>               设置IP端口"
            echo "  -gnpus <num>                    设置NPU数量"
            echo "  -fnpu <id>                      设置首个NPU ID"
            echo "  -fpe <id>                       设置首个PE ID"
            echo "  -a|--analyse <none|plot|md>    设置分析模式 (none=不生成, plot=只生成图, md=同时生成图和md), 默认none"
            exit 1
            ;;
    esac
done

# 验证测试类型
VALID_TEST_TYPES="put bi_put get bi_get all"
if [[ ! " $VALID_TEST_TYPES " =~ " $TEST_TYPE " ]]; then
    echo "错误: 测试类型必须是 'put'、'bi_put'、'get'、'bi_get' 或 'all'"
    exit 1
fi

# 验证数据类型
VALID_DATATYPES="float int8 int16 int32 int64 uint8 uint16 uint32 uint64 char all"
if [[ ! " $VALID_DATATYPES " =~ " $DATA_TYPE " ]]; then
    echo "错误: 数据类型必须是 'float'、'int8'、'int16'、'int32'、'int64'、'uint8'、'uint16'、'uint32'、'uint64'、'char' 或 'all'"
    exit 1
fi

VALID_MEMORY_TYPES="hbm dram"
if [[ ! " $VALID_MEMORY_TYPES " =~ " $MEMORY_TYPE " ]]; then
    echo "错误: SHMEM内存类型必须是 'hbm' 或 'dram'"
    exit 1
fi

echo "测试类型: $TEST_TYPE"
echo "数据类型: $DATA_TYPE"
echo "核数范围: $MIN_BLOCK_SIZE-$MAX_BLOCK_SIZE"
echo "幂数范围: $MIN_EXPONENT-$MAX_EXPONENT"
echo "循环次数: $LOOP_COUNT"
echo "UB size(KB): $UB_SIZE"
echo "SHMEM内存类型: $MEMORY_TYPE"
echo "PE_SIZE: $PE_SIZE, GNPU_NUM: $GNPU_NUM"
echo "FIRST_NPU: $FIRST_NPU, FIRST_PE: $FIRST_PE"

export SHMEM_CYCLE_PROF_PE=${SHMEM_CYCLE_PROF_PE:-0}

ALL_TEST_TYPES=("put" "bi_put" "get" "bi_get")
ALL_DATATYPES=("float" "int8" "int16" "int32" "int64" "uint8" "uint16" "uint32" "uint64" "char")

run_test() {
    local test_type="$1"
    local data_type="$2"
    for (( idx =0; idx < ${GNPU_NUM}; idx = idx + 1 )); do
        ${EXEC_BIN} --pes "$PE_SIZE" --pe-id "$idx" --ipport "$IPPORT" --gnpus "$GNPU_NUM" --fpe "$FIRST_PE" --fnpu "$FIRST_NPU" -t "$test_type" -d "$data_type" --block-range "$MIN_BLOCK_SIZE" "$MAX_BLOCK_SIZE" --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" --ub-size "$UB_SIZE" --memory-type "$MEMORY_TYPE" &
    done
    wait
}

if [[ "$TEST_TYPE" == "all" && "$DATA_TYPE" == "all" ]]; then
    echo "正在运行所有测试类型和数据类型..."
    for type in "${ALL_TEST_TYPES[@]}"; do
        for dtype in "${ALL_DATATYPES[@]}"; do
            echo "
=== 运行测试类型: $type, 数据类型: $dtype ==="
            run_test "$type" "$dtype"
        done
    done
elif [[ "$TEST_TYPE" == "all" ]]; then
    echo "正在运行所有测试类型，数据类型: $DATA_TYPE..."
    for type in "${ALL_TEST_TYPES[@]}"; do
        echo "
=== 运行测试类型: $type, 数据类型: $DATA_TYPE ==="
        run_test "$type" "$DATA_TYPE"
    done
elif [[ "$DATA_TYPE" == "all" ]]; then
    echo "正在运行测试类型: $TEST_TYPE，所有数据类型..."
    for dtype in "${ALL_DATATYPES[@]}"; do
        echo "
=== 运行测试类型: $TEST_TYPE, 数据类型: $dtype ==="
        run_test "$TEST_TYPE" "$dtype"
    done
else
    run_test "$TEST_TYPE" "$DATA_TYPE"
fi

cd ${CURRENT_DIR}

# 运行 perf_data_process.py 处理 output 目录
PERF_SCRIPT="${SCRIPT_DIR}/../../utils/perf_data_process.py"
if [ -f "${PERF_SCRIPT}" ]; then
    if [ "$ANALYSE_MODE" = "plot" ] || [ "$ANALYSE_MODE" = "md" ]; then
        echo -e "\n========== Generating performance charts =========="
        cmd_args=("-d" "${SCRIPT_DIR}/output")
        if [ "$ANALYSE_MODE" = "md" ]; then
            # md模式，图和md都生成
            true
        else
            # plot模式，只生成图，不生成md
            cmd_args+=("--no-markdown")
        fi
        python3 "${PERF_SCRIPT}" "${cmd_args[@]}"
    fi
fi
