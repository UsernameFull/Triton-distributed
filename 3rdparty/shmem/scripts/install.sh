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
install_flag=n
quiet_flag=n
uninstall_flag=n
install_path_flag=n
install_for_all_flag=n
sourcedir=$PWD
default_install_dir="/usr/local/Ascend/shmem"
VERSION=VERSION_PLACEHOLDER

function print()
{
    echo "[${1}] ${2}"
}

function chmod_authority()
{
    chmod_file ${default_install_dir}
    chmod_file ${install_dir}
    local file_rights=$([ "${install_for_all_flag}" == "y" ] && echo 555 || echo 550)
    chmod ${file_rights} ${install_dir}/scripts/uninstall.sh
    chmod ${file_rights} ${install_dir}/install.sh
    chmod_dir ${default_install_dir} "550"
    chmod_dir ${install_dir} "550"
    local path_rights=$([ "${install_for_all_flag}" == "y" ] && echo 755 || echo 750)
    chmod ${path_rights} ${default_install_dir}
    chmod ${path_rights} ${install_dir}
}

function chmod_file()
{
    chmod_recursion ${1} "550" "file" "*.sh"
    chmod_recursion ${1} "440" "file" "*.bin"
    chmod_recursion ${1} "440" "file" "*.h"
    chmod_recursion ${1} "440" "file" "*.info"
    chmod_recursion ${1} "440" "file" "*.so"
    chmod_recursion ${1} "440" "file" "*.a"
}

function chmod_dir()
{
    chmod_recursion ${1} ${2} "dir" 
}

function chmod_recursion()
{
    local parameter2=$2
    local rights="$(echo ${parameter2:0:2})""$(echo ${parameter2:1:1})"
    rights=$([ "${install_for_all_flag}" == "y" ] && echo ${rights} || echo $2)
    if [ "$3" = "dir" ]; then
        find $1 -type d -exec chmod ${rights} {} \; 2>/dev/null
    elif [ "$3" = "file" ]; then
        find $1 -type f -name "$4" -exec chmod ${rights} {} \; 2>/dev/null
    fi
}

function parse_script_args()
{
    while true
    do
        case "$1" in
        --install)
            install_flag=y
            shift
        ;;
        --quiet)
            quiet_flag=y
            shift
        ;;
        --install-path=*)
            install_path_flag=y
            target_dir=$(echo $1 | cut -d"=" -f2-)
            target_dir=${target_dir}/shmem
            shift
        ;;
        --uninstall)
            uninstall_flag=y
            shift
        ;;
        --install-for-all)
            install_for_all_flag=y
            shift
        ;;
        --*)
            shift
        ;;
        *)
            break
        ;;
        esac
    done
}

function check_owner()
{
    local cur_owner=$(whoami)

    if [ "${ASCEND_TOOLKIT_HOME}" == "" ]; then
        print "ERROR" "please check env ASCEND_TOOLKIT_HOME is set"
        exit 1
    fi

    if [ "${ASCEND_HOME_PATH}" == "" ]; then
        print "ERROR" "please check env ASCEND_HOME_PATH is set"
        exit 1
    else
        cann_path=${ASCEND_HOME_PATH}
    fi

    if [ ! -d "${cann_path}" ]; then
        print "ERROR" "can not find ${cann_path}"
        exit 1
    fi

    cann_owner=$(stat -c %U "${cann_path}")
    if [ "${cann_owner}" != "${cur_owner}" ]; then
        print "ERROR" "cur_owner is not same with CANN"
        exit 1
    fi

    if [[ "${cur_owner}" != "root" && "${install_flag}" == "y" ]]; then
        default_install_dir="${HOME}/Ascend/shmem"
    fi
    
    if [ "${install_path_flag}" == "y" ]; then
        default_install_dir="${target_dir}"
    fi
    print "INFO" "Check owner success"

}

function delete_latest()
{
    cd $default_install_dir
    print "INFO" "SHMEM delete latest!"
    if [ -d "latest" ]; then
        rm -rf latest
    fi
    if [ -f "set_env.sh" ]; then
        rm -rf set_env.sh
    fi
}

function delete_install_files()
{
    install_dir=$1
    print "INFO" "SHMEM $(basename $1) delete install files!"
    [ -n "$1" ] && rm -rf $1
}

function delete_file_with_authority()
{
    file_path=$1
    dir_path=$(dirname ${file_path})
    if [ ${dir_path} != "." ]; then
        dir_authority=$(stat -c %a ${dir_path})
        chmod 700 ${dir_path}
        if [ -d ${file_path} ]; then
            rm -rf ${file_path}
        else
            rm -f ${file_path}
        fi
        chmod ${dir_authority} ${dir_path}
    else
        chmod 700 ${file_path}
        if [ -d ${file_path} ]; then
            rm -rf ${file_path}
        else
            rm -f ${file_path}
        fi
    fi
}

function delete_empty_recursion()
{
    if [ ! -d $1 ]; then
        return 0
    fi
    print "INFO" "SHMEM $(basename $1) delete empty recursion!"
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
    if [ "$2" == "y" -a -z "$(ls $shmem_dir)" ]; then
        rm -rf $shmem_dir
    fi
    print "INFO" "SHMEM $(basename $1) uninstall success!"
}

function uninstall()
{
    install_dir=${default_install_dir}/${VERSION}
    uninstall_process ${install_dir} y
}

function check_target_dir_owner()
{
    local cur_owner=$(whoami)
    if [ "$cur_owner" != "root" ]; then
        return
    fi
    local seg_num=$(expr $(echo ${1} | grep -o "/" | wc -l) + "1")
    local path=""
    for((i=1;i<=$seg_num;i++))
    do
        local split=$(echo ${1} | cut -d "/" -f$i)
        if [ "$split" == "" ]; then
            continue
        fi
        local path=${path}"/"${split}
        if [ -d "$path" ]; then
            local path_owner=$(stat -c %U "${path}")
            if [ "$path_owner" != "root" ]; then
                print "ERROR" "Install failed, install path or its parents path $path owner [$path_owner] is inconsistent with current user [$cur_owner]"
                exit 1
            fi
        fi
    done
}

function check_path()
{
    if [ ! -d ${install_dir} ]; then
        mkdir -p ${install_dir}
        if [ ! -d ${install_dir} ]; then
            print "ERROR" "Install failed, create ${install_dir} failed"
            exit 1
        fi
    fi
}

function install_to_path()
{
    install_dir=${default_install_dir}/${VERSION}
    if [ -d ${install_dir} ]; then
        print "INFO" "The installation directory exists, uninstall first"
    fi
    uninstall_process ${install_dir}
    check_target_dir_owner
    check_path
    cd ${install_dir}
    copy_files

    cd ${default_install_dir}
    ln -snf $VERSION latest

}
function copy_files()
{
    mkdir -p $install_dir
    cp -r ${sourcedir}/* $install_dir
}
function install_process()
{
    local arch_pkg=SHMEMPKGARCH

    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        echo "it is system of x86_64"
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        echo "it is system of aarch64"
        ARCH="aarch64"
    fi

    if [ -n "${ARCH}" ]; then
        if [ "${arch_pkg}" != "${ARCH}" ]; then
            print "ERROR" "Install failed, pkg arch ${arch_pkg} is not consistent with the current environment ${ARCH}"
            exit 1
        fi
    fi

    if [ -n "${target_dir}" ]; then
        if [[ ! "${target_dir}" = /* ]]; then
            print "ERROR" "Install failed, [ERROR] use absolute  path for --install-path argument"
            exit 1
        fi
        install_to_path
    else
        install_to_path
    fi
}

function main()
{
    parse_script_args $*
    if [ "$uninstall_flag" == "y" ]; then
        uninstall
    elif [ "$install_flag" == "y" ] || [ "$install_path_flag" == "y" ]; then
        check_owner
        install_process
        chmod_authority
        print "INFO" "SHMEM install success"
    fi
}

main $*