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
readonly SCRIPT_DIR=$(dirname $(readlink -f "$0"))
readonly PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
readonly EXAMPLES_DIR=${PROJECT_ROOT}/examples/
readonly BUILD_DIR=${PROJECT_ROOT}/build/

exec_name=$1

function run_allgather()
{
    echo "begin run allgather"
    if [[ ! -f ${BUILD_DIR}/bin/allgather ]]; then
        echo "allgather build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/allgather/
    cd ${cur_dir}
    bash run.sh -pes 2 -type int
    return $?
}

function run_allgather_matmul()
{
    echo "begin run allgather_matmul"
    if [[ ! -f ${BUILD_DIR}/bin/allgather_matmul ]]; then
        echo "allgather_matmul build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/allgather_matmul/
    cd ${cur_dir}
    bash scripts/run.sh 0,1
    return $?
}

function run_allgather_matmul_padding()
{
    echo "begin run allgather_matmul_paddding"
    if [[ ! -f ${BUILD_DIR}/bin/allgather_matmul_padding ]]; then
        echo "allgather_matmul_padding build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/allgather_matmul_padding/
    cd ${cur_dir}
    bash scripts/run.sh 0,1
    return $?
}

function run_allgather_matmul_with_gather_result()
{
    echo "begin run allgather_matmul_with_gather_result"
    if [[ ! -f ${BUILD_DIR}/bin/allgather_matmul_with_gather_result ]]; then
        echo "allgather_matmul_with_gather_result build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/allgather_matmul_with_gather_result/
    cd ${cur_dir}
    bash scripts/run.sh 0,1
    return $?
}

function run_dispatch_gmm_combine()
{
    echo "begin run dispatch_gmm_combine"
    if [[ ! -f ${BUILD_DIR}/bin/dispatch_gmm_combine ]]; then
        echo "dispatch_gmm_combine build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/dispatch_gmm_combine/
    cd ${cur_dir}
    bash scripts/run.sh -pes 2 -M 64 -K 7168 -N 4096 -expertPerPe 2 -dataType 2 -weightNz 1 -transB 0
    return $?
}

function run_dynamic_tiling()
{
    echo "begin run dynamic_tiling"
    if [[ ! -f ${BUILD_DIR}/bin/dynamic_tiling ]]; then
        echo "dynamic_tiling build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/dynamic_tiling/
    cd ${cur_dir}
    bash scripts/run.sh 0 1 0,1
    return $?
}

function run_kv_shuffle()
{
    echo "begin run kv_shuffle"
    if [[ ! -f ${BUILD_DIR}/bin/kv_shuffle ]]; then
        echo "kv_shuffle build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/kv_shuffle/
    cd ${cur_dir}
    bash scripts/run.sh 2
    return $?
}

function run_matmul_allreduce()
{
    echo "begin run matmul_allreduce"
    if [[ ! -f ${BUILD_DIR}/bin/matmul_allreduce ]]; then
        echo "matmul_allreduce build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/matmul_allreduce/
    cd ${cur_dir}
    bash scripts/run.sh 0,1
    return $?
}

function run_matmul_reduce_scatter()
{
    echo "begin run matmul_reduce_scatter"
    if [[ ! -f ${BUILD_DIR}/bin/matmul_reduce_scatter ]]; then
        echo "matmul_reduce_scatter build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/matmul_reduce_scatter/
    cd ${cur_dir}
    bash scripts/run.sh 0,1
    return $?
}

function run_matmul_reduce_scatter_padding()
{
    echo "begin run matmul_reduce_scatter_padding"
    if [[ ! -f ${BUILD_DIR}/bin/matmul_reduce_scatter_padding ]]; then
        echo "matmul_reduce_scatter_padding build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/matmul_reduce_scatter_padding/
    cd ${cur_dir}
    bash scripts/run.sh 0,1
    return $?
}

function run_rdma_demo()
{
    echo "begin run rdma_demo"
    if [[ ! -f ${BUILD_DIR}/bin/rdma_demo ]]; then
        echo "rdma_demo build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/rdma_demo/
    cd ${cur_dir}
    bash run.sh
    return $?
}

function run_unuse_handlewait()
{
    echo "begin run unuse_handlewait"
    if [[ ! -f ${BUILD_DIR}/bin/unuse_handlewait ]]; then
        echo "unuse_handlewait build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/rdma_handlewait_test/unuse_handlewait/
    cd ${cur_dir}
    bash run.sh
    return $?
}

function run_use_handlewait()
{
    echo "begin run use_handlewait"
    if [[ ! -f ${BUILD_DIR}/bin/use_handlewait ]]; then
        echo "use_handlewait build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/rdma_handlewait_test/use_handlewait/
    cd ${cur_dir}
    bash run.sh
    return $?
}

function run_rdma_perftest()
{
    echo "begin run rdma_perftest"
    if [[ ! -f ${BUILD_DIR}/bin/rdma_perftest ]]; then
        echo "rdma_perftest build output doest exist. Execute 'script/build.sh -examples' first"
        return 1
    fi

    cur_dir=${EXAMPLES_DIR}/rdma_perftest/
    cd ${cur_dir}
    bash run.sh
    return $?
}

function run_python_extesion()
{
    echo "begin run python extension"
    cur_dir=${EXAMPLES_DIR}/python_extension/
    cd $cur_dir
    bash run.sh
    return $?
}

function run_all()
{
    run_allgather || return 1
    run_allgather_matmul || return 2
    run_allgather_matmul_padding || return 3
    run_allgather_matmul_with_gather_result || return 4
    #run_dispatch_gmm_combine || return 5
    run_dynamic_tiling || return 6
    run_kv_shuffle || return 7
    run_matmul_allreduce || return 8
    run_matmul_reduce_scatter || return 9
    run_matmul_reduce_scatter_padding || return 10
    # run_rdma_demo || return 11
    # run_unuse_handlewait || return 12
    # run_use_handlewait || return 13
    # run_rdma_perftest || return 14
    #run_python_extesion || return 15
    return 0
}

function main()
{
    case $exec_name in
        "")
            run_all
            ;;
        allgather)
            run_allgather || return 1
            ;;
        allgather_matmul)
            run_allgather_matmul || return 2
            ;;
        allgather_matmul_padding)
            run_allgather_matmul_padding || return 3
            ;;
        allgather_matmul_with_gather_result)
            run_allgather_matmul_with_gather_result || return 4
            ;;
        dispatch_gmm_combine)
            run_dispatch_gmm_combine || return 5
            ;;
        dynamic_tiling)
            run_dynamic_tiling || return 6
            ;;
        kv_shuffle)
            run_kv_shuffle || return 7
            ;;
        matmul_allreduce)
            run_matmul_allreduce || return 8
            ;;
        matmul_reduce_scatter)
            run_matmul_reduce_scatter || return 9
            ;;
        matmul_reduce_scatter_padding)
            run_matmul_reduce_scatter_padding || return 10
            ;;
        rdma_demo)  # not ready
            run_rdma_demo || return 11
            ;;
        unuse_handlewait)
            run_unuse_handlewait || return 12
            ;;
        use_handlewait)     # not ready
            run_use_handlewait || return 13
            ;;
        rdma_perftest)  # not ready
            run_rdma_perftest || return 14
            ;;
        python_extension)
            run_python_extesion || return 15
            ;;
        *)
            echo "unknown example name: ${exec_name}"
            ;;
    esac
    return 0
}

main
ret=$?
if [[ $ret -ne 0 ]]; then
    echo "run example failed return ${ret}"
else
    echo "run example finished"
fi
exit $ret