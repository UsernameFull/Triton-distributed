#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should have received a copy of the License along with this program.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))

# 默认测试类型为put
TEST_TYPE="put"
# 默认数据类型为float
DATA_TYPE="float"
# 默认核数范围
MIN_BLOCK_SIZE="32"
MAX_BLOCK_SIZE="32"
# 默认幂数范围
MIN_EXPONENT="3"
MAX_EXPONENT="20"
# 默认循环次数
LOOP_COUNT="1000"
# 默认UB size(KB)
UB_SIZE="16"
# 默认SHMEM内存类型: hbm/dram，仅shmem模式使用
MEMORY_TYPE="hbm"
# 默认运行模式: ascendc/shmem/all (all表示全跑)
MODE="all"
# 分析模式: none/plot/md (none表示都不生成, plot表示只生成图, md表示同时生成图和md), 默认none
ANALYSE_MODE="none"
# 默认RANK配置
PE_SIZE="2"
IPPORT="tcp://127.0.0.1:8760"
GNPU_NUM="2"
FIRST_NPU="0"
FIRST_PE="0"

function usage() {
    echo "Usage: bash $0 [options]"
    echo "Options:"
    echo "  -h|--help                 Show this help message"
    echo "  -t|--test-type <type>      设置测试类型 (put|get|ub2gm_local|ub2gm_remote|gm2ub_local|gm2ub_remote|all)"
    echo "  -d|--datatype <type>      设置数据类型 (float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char|all)"
    echo "  -b|--block-size <size>          设置核数"
    echo "  --block-range <min> <max>       设置核数范围"
    echo "  -e|--exponent <exponent>        设置数据量的幂数"
    echo "  --exponent-range <min> <max>    设置数据量的幂数范围"
    echo "  --loop-count <count>            设置循环次数"
    echo "  --ub-size <size>                设置UB size(KB), 默认16"
    echo "  --memory-type <hbm|dram>        设置SHMEM内存类型, 默认hbm"
    echo "  -pes <size>                     设置PE大小"
    echo "  -ipport <ip:port>               设置IP端口"
    echo "  -gnpus <num>                   设置NPU数量"
    echo "  -fnpu <id>                     设置首个NPU ID"
    echo "  -fpe <id>                      设置首个PE ID"
    echo "  -m|--mode <ascendc|shmem|all>  设置运行模式 (ascendc=只跑ascendc, shmem=只跑shmem, all=全跑), 默认all"
    echo "  -a|--analyse <none|plot|md>    设置分析模式 (none=不生成, plot=只生成图, md=同时生成图和md), 默认none"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 1
            ;;
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
        -pes)
            if [ -n "$2" ]; then
                PE_SIZE="$2"
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
        -m|--mode)
            if [ -n "$2" ]; then
                MODE="$2"
                shift 2
            else
                echo "Error: -m|--mode requires a value."
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
            echo "Error: Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

VALID_MEMORY_TYPES="hbm dram"
if [[ ! " $VALID_MEMORY_TYPES " =~ " $MEMORY_TYPE " ]]; then
    echo "错误: SHMEM内存类型必须是 'hbm' 或 'dram'"
    exit 1
fi

echo "=============================================="
echo "example/mte_perftest"
echo "=============================================="
echo "测试类型: $TEST_TYPE"
echo "数据类型: $DATA_TYPE"
echo "核数范围: $MIN_BLOCK_SIZE-$MAX_BLOCK_SIZE"
echo "幂数范围: $MIN_EXPONENT-$MAX_EXPONENT"
echo "循环次数: $LOOP_COUNT"
echo "UB size(KB): $UB_SIZE"
echo "SHMEM内存类型: $MEMORY_TYPE"
echo "运行模式: $MODE"
echo "PE_SIZE: $PE_SIZE, GNPU_NUM: $GNPU_NUM"
echo "FIRST_NPU: $FIRST_NPU, FIRST_PE: $FIRST_PE"
echo "=============================================="

DEVICE1=$FIRST_NPU
DEVICE2=$((FIRST_NPU + 1))

# 清理之前的output
echo -e "\n========== Cleaning previous outputs =========="
if [[ "$MODE" == "ascendc" || "$MODE" == "all" ]]; then
    rm -rf "${SCRIPT_DIR}/ascendc_perftest/output"
    echo "Cleaned: ${SCRIPT_DIR}/ascendc_perftest/output"
fi
if [[ "$MODE" == "shmem" || "$MODE" == "all" ]]; then
    rm -rf "${SCRIPT_DIR}/shmem_perftest/output"
    echo "Cleaned: ${SCRIPT_DIR}/shmem_perftest/output"
fi

# 清理外层output目录
rm -rf "${SCRIPT_DIR}/output"
echo "Cleaned: ${SCRIPT_DIR}/output"

if [[ "$MODE" == "ascendc" || "$MODE" == "all" ]]; then
    echo -e "\n========== Running ascendc_perftest =========="
    echo "Command: bash ${SCRIPT_DIR}/ascendc_perftest/run.sh -t \"$TEST_TYPE\" -d \"$DATA_TYPE\" --block-range \"$MIN_BLOCK_SIZE\" \"$MAX_BLOCK_SIZE\" --exponent-range \"$MIN_EXPONENT\" \"$MAX_EXPONENT\" --loop-count \"$LOOP_COUNT\" --device1 \"$DEVICE1\" --device2 \"$DEVICE2\" --ub-size \"$UB_SIZE\""
    bash "${SCRIPT_DIR}/ascendc_perftest/run.sh" -t "$TEST_TYPE" -d "$DATA_TYPE" --block-range "$MIN_BLOCK_SIZE" "$MAX_BLOCK_SIZE" --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" --device1 "$DEVICE1" --device2 "$DEVICE2" --ub-size "$UB_SIZE"
fi

if [[ "$MODE" == "shmem" || "$MODE" == "all" ]]; then
    echo -e "\n========== Running shmem_perftest =========="
    echo "Command: bash ${SCRIPT_DIR}/shmem_perftest/run.sh -t \"$TEST_TYPE\" -d \"$DATA_TYPE\" --block-range \"$MIN_BLOCK_SIZE\" \"$MAX_BLOCK_SIZE\" --exponent-range \"$MIN_EXPONENT\" \"$MAX_EXPONENT\" --loop-count \"$LOOP_COUNT\" -pes \"$PE_SIZE\" -ipport \"$IPPORT\" -gnpus \"$GNPU_NUM\" -fnpu \"$FIRST_NPU\" -fpe \"$FIRST_PE\" --ub-size \"$UB_SIZE\" --memory-type \"$MEMORY_TYPE\""
    bash "${SCRIPT_DIR}/shmem_perftest/run.sh" -t "$TEST_TYPE" -d "$DATA_TYPE" --block-range "$MIN_BLOCK_SIZE" "$MAX_BLOCK_SIZE" --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" -pes "$PE_SIZE" -ipport "$IPPORT" -gnpus "$GNPU_NUM" -fnpu "$FIRST_NPU" -fpe "$FIRST_PE" --ub-size "$UB_SIZE" --memory-type "$MEMORY_TYPE"
fi

# 拷贝output到外层output目录
echo -e "\n========== Copying outputs =========="
mkdir -p "${SCRIPT_DIR}/output"

if [[ "$MODE" == "ascendc" || "$MODE" == "all" ]]; then
    if [ -d "${SCRIPT_DIR}/ascendc_perftest/output" ]; then
        mkdir -p "${SCRIPT_DIR}/output/ascendc_perftest"
        cp -r "${SCRIPT_DIR}/ascendc_perftest/output"/* "${SCRIPT_DIR}/output/ascendc_perftest/"
        echo "Copied: ascendc_perftest -> ${SCRIPT_DIR}/output/ascendc_perftest"
    fi
fi

if [[ "$MODE" == "shmem" || "$MODE" == "all" ]]; then
    if [ -d "${SCRIPT_DIR}/shmem_perftest/output" ]; then
        mkdir -p "${SCRIPT_DIR}/output/shmem_perftest"
        cp -r "${SCRIPT_DIR}/shmem_perftest/output"/* "${SCRIPT_DIR}/output/shmem_perftest/"
        echo "Copied: shmem_perftest -> ${SCRIPT_DIR}/output/shmem_perftest"
    fi
fi

echo -e "\n=============================================="
echo "All tests completed!"
echo "Output directory: ${SCRIPT_DIR}/output"
echo "=============================================="

# 运行 perf_data_process.py 分别处理 output 下的子目录
PERF_SCRIPT="${SCRIPT_DIR}/../utils/perf_data_process.py"
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
