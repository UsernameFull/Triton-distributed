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
VERSION=VERSION_PLACEHOLDER
CUR_DIR=$(dirname $(readlink -f $0))

function print()
{
    echo "[${1}] ${2}"
}

function delete_install_files()
{
    install_dir=$1
    print "INFO" "SHMEM $(basename $1) delete install files."
    [ -n "$1" ] && rm -rf $1
}

function delete_file_with_authority()
{
    file_path=$1
    dir_path=$(dirname ${file_path})
    if [ ${dir_path} == "." ]; then
        chmod 700 ${file_path}
        if [ -d ${file_path} ]; then
            rm -rf ${file_path}
        else
            rm -f ${file_path}
        fi
    else
        dir_authority=$(stat -c %a ${dir_path})
        chmod 700 ${dir_path}
        if [ -d ${file_path} ]; then
            rm -rf ${file_path}
        else
            rm -f ${file_path}
        fi
        chmod ${dir_authority} ${dir_path}
    fi
}

function delete_empty_recursion()
{
    if [ ! -d $1 ]; then
        return 0
    fi
    print "INFO" "SHMEM $(basename $1) delete empty recursion."
    for file in $1/*
    do
        if [ -d $file ]; then
            delete_empty_recursion $file
        fi
    done
    if [ -z "$(ls -A $1)" ]; then
        delete_file_with_authority $1
    fi
}

function delete_latest()
{
    cd $1/..
    print "INFO" "SHMEM delete latest!"
    if [ -d "latest" ]; then
        rm -rf latest
    fi
    if [ -f "set_env.sh" ]; then
        rm -rf set_env.sh
    fi
}

function uninstall_process()
{
    if [ ! -d $1 ]; then
        return 0
    fi
    print "INFO" "SHMEM $(basename $1) uninstall start!"
    shmem_dir=$(cd $1/..;pwd)
    delete_latest $1
    delete_install_files $1
    if [ -d $1 ]; then
        delete_empty_recursion $1
    fi
    if [ -z "$(ls $shmem_dir)" ]; then
        rm -rf $shmem_dir
    fi
    print "INFO" "SHMEM $(basename $1) uninstall success!"
}


install_dir=$(cd ${CUR_DIR}/../../${VERSION};pwd)
uninstall_process ${install_dir}