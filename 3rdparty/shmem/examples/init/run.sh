#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

# 初始化默认值
MODE="default"  # 默认模式为default
NUM_PROCESSES=2 # 默认进程数为2
FIRST_NPU="0"
FIRST_PE="0"
GNPU_NUM="8"
IPPORT="tcp://127.0.0.1:8666"
SESSION_ID="127.0.0.1:8666"
ONLY_BUILD=OFF
# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -mode)
            MODE="$2"
            shift 2
            ;;
        -pesize)
            NUM_PROCESSES="$2"
            if ! [[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: pesize must be a positive integer!"
                exit 1
            fi
            if [ "$GNPU_NUM" -gt "$NUM_PROCESSES" ]; then
                GNPU_NUM="$NUM_PROCESSES"
                echo "Because GNPU_NUM is greater than NUM_PROCESSES, GNPU_NUM is assigned the value of NUM_PROCESSES=${NUM_PROCESSES}."
            fi
            shift 2
            ;;
        -fpe)
            if [ -n "$2" ]; then
                if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                    echo "Error: -fpe requires a numeric value."
                    exit 1
                fi
                FIRST_PE="$2"
                shift 2
            else
                echo "Error: -fpe requires a value."
                exit 1
            fi
            ;;
        -ipport)
            if [ -n "$2" ]; then
                if [[ "$2" =~ ^[a-zA-Z0-9/:._-]+$ ]]; then
                    IPPORT="tcp://${2}"
                    SESSION_ID="${2}"
                    shift 2
                else
                    echo "Error: Invalid -ipport format, only alphanumeric and :/_- allowed"
                    exit 1
                fi
            else
                echo "Error: -ipport requires a value."
                exit 1
            fi
            ;;
        -gnpus)
            if [ -n "$2" ]; then
                if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                    echo "Error: -gnpus requires a numeric value."
                    exit 1
                fi
                GNPU_NUM="$2"
                shift 2
            else
                echo "Error: -gnpus requires a value."
                exit 1
            fi
            ;;
        -fnpu)
            if [ -n "$2" ]; then
                if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                    echo "Error: -fnpu requires a numeric value."
                    exit 1
                fi
                FIRST_NPU="$2"
                shift 2
            else
                echo "Error: -fnpu requires a value."
                exit 1
            fi
            ;;
        -build)
            ONLY_BUILD=ON
            shift
            ;;
        *)
            echo "Error: Unknown parameter '$1'"
            echo "Usage: $0 [-mode <default|mpi|uid|uid_multi>] [-pesize <num>]"
            exit 1
            ;;
    esac
done

case "$MODE" in
    default)
        MODE_ID=1
        ;;
    mpi)
        MODE_ID=2
        ;;
    uid)
        MODE_ID=3
        ;;
    uid_multi)
        MODE_ID=4
        ;;
    uid_default)
        MODE_ID=5
        ;;
    *)
        echo "Error: Invalid mode '$MODE'! Only 'default'/'mpi'/'uid'/'uid_multi'/'uid_default' are allowed"
        echo "Usage: $0 [-mode <default|mpi|uid|uid_multi|uid_default>] [-pesize <num>]"
        exit 1
        ;;
esac

BUILD_DIR="build"
EXECUTABLE_NAME="init_examples"

case "$MODE" in
    mpi|uid|default|uid_default)
    export SHMEM_UID_SESSION_ID=$SESSION_ID
esac

echo "=== Prepare build directory ==="
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"/*
fi
mkdir -p "$BUILD_DIR"

echo "=== Run CMake with RUN_MODE=${MODE_ID} (mode: ${MODE}) ==="
cd "$BUILD_DIR" || { echo "Error: Failed to enter build directory!"; exit 1; }
cmake -DRUN_MODE="${MODE_ID}" ..
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed!"
    exit 1
fi

echo "=== Compile executable (mode: ${MODE}, pesize: ${NUM_PROCESSES}) ==="
make -j$(nproc) "${EXECUTABLE_NAME}"
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi
cd ..

if [ "$ONLY_BUILD" == "ON" ]; then
    echo "Compile only, prepare for cross-machine tasks."
    exit 0
fi

echo "=== Launch executable (mode: ${MODE}, pesize: ${NUM_PROCESSES}) ==="


case "$MODE" in
    mpi|uid|uid_multi|uid_default)
        if [ -f "hostfile" ]; then
            echo "Found hostfile, run mpirun with -f hostfile"
            mpirun -f hostfile ./build/bin/"${EXECUTABLE_NAME}" "$GNPU_NUM"
        else
            echo "No hostfile found, run mpirun without hostfile"
            mpirun -np "$NUM_PROCESSES" ./build/bin/"${EXECUTABLE_NAME}" "$NUM_PROCESSES"
        fi
        if [ $? -ne 0 ]; then
            echo "=== Execution failed (mode: ${MODE}, pesize: ${NUM_PROCESSES})! ==="
            exit 1
        fi
        ;;
    default)
        pids=()
        for (( idx=0; idx < GNPU_NUM; idx++ )); do
            echo "Starting process $idx/$NUM_PROCESSES..."
            ./build/bin/"${EXECUTABLE_NAME}" "$idx" "$NUM_PROCESSES" "${IPPORT}" "$GNPU_NUM" "$FIRST_PE" "$FIRST_NPU" &
            pids+=($!)
        done
        all_success=true
        for pid in "${pids[@]}"; do
            wait "$pid"
            if [ $? -ne 0 ]; then
                all_success=false
            fi
        done
        echo "=== All processes completed ==="
        if [ "$all_success" = false ]; then
            echo "=== Execution failed (mode: ${MODE}, pesize: ${NUM_PROCESSES})! ==="
            exit 1
        fi
        ;;
esac

echo "=== Execution succeeded (mode: ${MODE}, pesize: ${NUM_PROCESSES})! ==="