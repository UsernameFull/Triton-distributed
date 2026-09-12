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
set -e

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
EXAMPLE_DIR=$(dirname "$SCRIPT_DIR")
GROUP_DIR=$(dirname "$EXAMPLE_DIR")
PROJECT_ROOT=$(dirname "$(dirname "$GROUP_DIR")")

cd "${EXAMPLE_DIR}"

PE_SIZE="2"
IPPORT="tcp://127.0.0.1:8767"
GNPU_NUM="2"
FIRST_NPU="0"
FIRST_PE="0"
TEST_TYPE="int32_t"
BS="8"
# Fixed token hidden size (7k).
H="7168"
TOPK="8"
EXPERT_PER_PE="2"
PERF_MODE="0"
WARMUP_COUNT="5"
LOOP_COUNT="50"
BS_LIST=""
H_LIST=""
PES_LIST=""
TOPK_LIST=""
EXPERT_PER_PE_LIST=""
OUTPUT_DIR="output/perf"
PROF_PE="0"
ANALYSE_MODE="none"

split_list() {
    local value="$1"
    local fallback="$2"
    if [[ -z "${value}" ]]; then
        value="${fallback}"
    fi
    echo "${value//,/ }"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -pes) PE_SIZE="$2"; GNPU_NUM="$2"; shift 2 ;;
        -ipport) IPPORT="$2"; shift 2 ;;
        -gnpus) GNPU_NUM="$2"; shift 2 ;;
        -fnpu) FIRST_NPU="$2"; shift 2 ;;
        -fpe) FIRST_PE="$2"; shift 2 ;;
        -type) TEST_TYPE="$2"; shift 2 ;;
        -bs) BS="$2"; shift 2 ;;
        -h) H="$2"; shift 2 ;;
        -topk) TOPK="$2"; shift 2 ;;
        -expertPerPe) EXPERT_PER_PE="$2"; shift 2 ;;
        --perf) PERF_MODE="1"; shift ;;
        --warmup) WARMUP_COUNT="$2"; shift 2 ;;
        --loops) LOOP_COUNT="$2"; shift 2 ;;
        --bs-list) BS_LIST="$2"; shift 2 ;;
        --h-list) H_LIST="$2"; shift 2 ;;
        --pes-list) PES_LIST="$2"; shift 2 ;;
        --topk-list) TOPK_LIST="$2"; shift 2 ;;
        --expert-per-pe-list) EXPERT_PER_PE_LIST="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --prof-pe) PROF_PE="$2"; shift 2 ;;
        -a|--analyse) ANALYSE_MODE="$2"; shift 2 ;;
        *) echo "Error: Unknown option $1."; exit 1 ;;
    esac
done

if [[ "${TEST_TYPE}" == "bfloat16_t" ]]; then
    echo "Error: combine does not support bfloat16_t on this CANN backend because scalar bf16 casts are unsupported."
    exit 1
fi

export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
CASE_INDEX=0

run_one_case() {
    local pes="$1"
    local bs="$2"
    local h="$3"
    local topk="$4"
    local expert_per_pe="$5"
    local prof_pe_value="$6"

    if [[ "${GNPU_NUM}" -ne "${pes}" ]]; then
        GNPU_NUM="${pes}"
    fi

    local moe_expert_num=$((pes * expert_per_pe))
    local case_id="shape_${bs}_${h}_${topk}_${moe_expert_num}_${pes}"
    local csv_path="${OUTPUT_DIR}/combine_perf_rank${prof_pe_value}.csv"
    local case_port=$((8767 + CASE_INDEX % 1000))
    local session_port=$((8898 + CASE_INDEX % 1000))
    local run_ipport="${IPPORT}"
    if [[ "${IPPORT}" == tcp://127.0.0.1:* ]]; then
        run_ipport="tcp://127.0.0.1:${case_port}"
    fi
    export SHMEM_UID_SESSION_ID="127.0.0.1:${session_port}"
    CASE_INDEX=$((CASE_INDEX + 1))

    echo "[Combine] Running ${case_id}, prof_pe=${prof_pe_value}, ipport=${run_ipport}, session=${SHMEM_UID_SESSION_ID}"
    mkdir -p output "${OUTPUT_DIR}"
    rm -f output/x_out_*.bin
    python3 "${GROUP_DIR}/scripts/data_gen.py" --pes "${pes}" --bs "${bs}" --h "${h}" --topk "${topk}" \
        --expert-per-pe "${expert_per_pe}" --dtype "${TEST_TYPE}"

    if [[ "${PERF_MODE}" == "1" ]]; then
        export SHMEM_CYCLE_PROF_PE="${prof_pe_value}"
    else
        unset SHMEM_CYCLE_PROF_PE
    fi

    pids=()
    for ((idx = 0; idx < pes; idx++)); do
        ${PROJECT_ROOT}/build/bin/combine "${pes}" "${idx}" "${run_ipport}" "${pes}" "${FIRST_PE}" "${FIRST_NPU}" \
            "${TEST_TYPE}" "${bs}" "${h}" "${topk}" "${expert_per_pe}" "${PERF_MODE}" "${WARMUP_COUNT}" \
            "${LOOP_COUNT}" "${csv_path}" "${case_id}" &
        pids+=("$!")
    done

    ret=0
    for pid in "${pids[@]}"; do
        wait "$pid" || ret=$?
    done
    if [[ "${ret}" -ne 0 ]]; then
        return "${ret}"
    fi

    python3 "${GROUP_DIR}/scripts/check_combine.py" --pes "${pes}" --bs "${bs}" --h "${h}" --topk "${topk}" \
        --expert-per-pe "${expert_per_pe}" --dtype "${TEST_TYPE}"
}

rm -rf golden output
mkdir -p output

ret=0
if [[ "${PERF_MODE}" == "1" ]]; then
    for pes in $(split_list "${PES_LIST}" "${PE_SIZE}"); do
        for bs in $(split_list "${BS_LIST}" "${BS}"); do
            for h_value in $(split_list "${H_LIST}" "${H}"); do
                for topk_value in $(split_list "${TOPK_LIST}" "${TOPK}"); do
                    for expert_per_pe_value in $(split_list "${EXPERT_PER_PE_LIST}" "${EXPERT_PER_PE}"); do
                        if [[ "${PROF_PE}" == "all" ]]; then
                            for ((prof_idx = 0; prof_idx < pes; prof_idx++)); do
                                run_one_case "${pes}" "${bs}" "${h_value}" "${topk_value}" \
                                    "${expert_per_pe_value}" "${prof_idx}" || ret=$?
                                if [[ "${ret}" -ne 0 ]]; then
                                    cd "${CURRENT_DIR}"
                                    exit "${ret}"
                                fi
                            done
                        else
                            run_one_case "${pes}" "${bs}" "${h_value}" "${topk_value}" \
                                "${expert_per_pe_value}" "${PROF_PE}" || ret=$?
                            if [[ "${ret}" -ne 0 ]]; then
                                cd "${CURRENT_DIR}"
                                exit "${ret}"
                            fi
                        fi
                    done
                done
            done
        done
    done
else
    if [[ "${GNPU_NUM}" -ne "${PE_SIZE}" ]]; then
        echo "Error: combine run.sh expects -gnpus to equal -pes for this single-node example."
        exit 1
    fi
    run_one_case "${PE_SIZE}" "${BS}" "${H}" "${TOPK}" "${EXPERT_PER_PE}" "0" || ret=$?
fi

if [[ "${PERF_MODE}" == "1" && "${PROF_PE}" == "all" ]]; then
    python3 "${PROJECT_ROOT}/examples/utils/summarize_moe_perf.py" --dir "${OUTPUT_DIR}" \
        --output "combine_perf_summary.csv"
fi

PERF_SCRIPT="${PROJECT_ROOT}/examples/utils/perf_data_process.py"
if [[ "${PERF_MODE}" == "1" && -f "${PERF_SCRIPT}" ]]; then
    if [[ "${ANALYSE_MODE}" == "plot" || "${ANALYSE_MODE}" == "md" ]]; then
        cmd_args=("-d" "${EXAMPLE_DIR}/${OUTPUT_DIR}")
        if [[ "${ANALYSE_MODE}" == "plot" ]]; then
            cmd_args+=("--no-markdown")
        fi
        python3 "${PERF_SCRIPT}" "${cmd_args[@]}"
    fi
fi

cd "${CURRENT_DIR}"
exit "${ret}"
