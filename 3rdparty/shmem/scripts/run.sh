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
readonly CURRENT_DIR=$(pwd)
readonly SCRIPT_DIR=$(dirname $(readlink -f "${BASH_SOURCE[0]}"))
readonly PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
readonly BUILD_PATH="$PROJECT_ROOT/build"
readonly COVERAGE_PATH="$BUILD_PATH/coverage"

lcov --version > /dev/null 2>&1
lcov_not_found=$?

cd ${PROJECT_ROOT}
rm -rf "$COVERAGE_PATH"

set -e
RANK_SIZE="8"
IPPORT="tcp://127.0.0.1:8666"
SESSION_ID="127.0.0.1:8766"
GNPU_NUM="8"
FIRST_NPU="0"
FIRST_RANK="0"
if [ -z "${GTEST_FILTER}" ]; then
    TEST_FILTER="*.*"
else
    TEST_FILTER="${GTEST_FILTER}"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -ranks)
            if [ -n "$2" ]; then
                if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                    echo "Error: -ranks requires a numeric value."
                    exit 1
                fi
                RANK_SIZE="$2"
                if [ "$GNPU_NUM" -gt "$RANK_SIZE" ]; then
                    GNPU_NUM="$RANK_SIZE"
                    echo "Because GNPU_NUM is greater than RANK_SIZE, GNPU_NUM is assigned the value of RANK_SIZE=${RANK_SIZE}."
                fi
                shift 2
            else
                echo "Error: -ranks requires a value."
                exit 1
            fi
            ;;
        -frank)
            if [ -n "$2" ]; then
                if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                    echo "Error: -frank requires a numeric value."
                    exit 1
                fi
                FIRST_RANK="$2"
                shift 2
            else
                echo "Error: -frank requires a value."
                exit 1
            fi
            ;;
        -ipport)
            if [ -n "$2" ]; then
                if [[ "$2" =~ ^[a-zA-Z0-9/:._-]+$ ]]; then
                    IPPORT="tcp://${2}"
                    SESSION_ID="${2}"
                    export SHMEM_UID_SESSION_ID=$SESSION_ID
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
        -test_filter)
            if [ -n "$2" ]; then
                FILTERED_VALUE=$(echo "$2" | sed 's/[;&|]*//g')
                if [ -z "$FILTERED_VALUE" ]; then
                    echo "Invalid test_filter value"
                    exit 1
                fi
                ESCAPED_VALUE=$(printf '%q' "$FILTERED_VALUE")
                TEST_FILTER="*$ESCAPED_VALUE*"
                shift 2
            else
                echo "Error: -test_filter requires a value."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done
export SMEM_CONF_STORE_TLS_ENABLE=0
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

# Run unit test
cd "$BUILD_PATH"
./bin/aclshmem_unittest "$RANK_SIZE" "$IPPORT" "$GNPU_NUM" "$FIRST_RANK" "$FIRST_NPU"  --gtest_output=xml:test_detail.xml --gtest_filter=${TEST_FILTER}

# Collect coverage
if [[ $lcov_not_found -ne 0 ]]; then
    echo "lcov not found, code coverage generation will skipped."
else
    mkdir -p "$COVERAGE_PATH"
    cd "$COVERAGE_PATH"
    lcov -d "$BUILD_PATH" --c --output-file "$COVERAGE_PATH/coverage.info" -rc lcov_branch_coverage=1 --rc lcov_excl_br_line="LCOV_EXCL_BR_LINE|SHM_LOG_*|SHM_ASSERT*|ACLSHMEM_CHECK_RET"
    lcov -e "$COVERAGE_PATH/coverage.info" "*/src/host/*" -o "$COVERAGE_PATH/coverage.info" --rc lcov_branch_coverage=1
    lcov -r "$COVERAGE_PATH/coverage.info" "*src/host/common/*" -o "$COVERAGE_PATH/coverage.info" --rc lcov_branch_coverage=1
    genhtml -o "$COVERAGE_PATH/result" "$COVERAGE_PATH/coverage.info" --show-details --legend --rc lcov_branch_coverage=1
fi

cd ${CURRENT_DIR}
