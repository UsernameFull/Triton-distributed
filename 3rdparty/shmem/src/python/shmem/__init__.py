#!/usr/bin/env python
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#

import os
import sys
import ctypes
import logging
import torch
import torch_npu

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
sys.path.append(current_dir)
libs_path = current_dir

# List of required .so files (add new dependencies here)
required_so_files = [
    "libshmem_utils.so",
    "aclshmem_bootstrap_config_store.so",
    "libshmem.so",
]

for lib in required_so_files:
    lib_path = os.path.join(libs_path, lib)
    try:
        ctypes.CDLL(lib_path)
    except FileNotFoundError:
        raise RuntimeError(f"Shared library file not found: {lib_path}")
    except OSError as e:
        raise RuntimeError(f"Failed to load shared library {lib_path}: {e}")

from . import core
from .construct_tensor import calc_nbytes, construct_tensor_from_ptr
from ._pyshmem import (aclshmem_init, aclshmem_get_unique_id, aclshmem_init_using_unique_id, \
                       aclshmem_finalize, aclshmem_malloc, aclshmem_free, \
                       aclshmem_ptr, my_pe, pe_count, set_conf_store_tls_key, team_split_strided, \
                       team_split_2d, team_translate_pe, \
                       team_destroy, InitAttr, OpEngineType, \
                       InitStatus, aclshmem_calloc, aclshmem_align, aclshmemx_init_status, get_ffts_config, \
                       team_my_pe, team_n_pes, \
                       aclshmem_putmem_nbi, aclshmem_getmem_nbi, aclshmem_putmem, aclshmem_getmem, \
                       aclshmemx_putmem_signal, \
                       aclshmemx_putmem_signal_nbi, \
                       aclshmem_info_get_version, aclshmem_info_get_name, \
                       aclshmem_team_get_config, OptionalAttr, aclshmem_global_exit, set_conf_store_tls, \
                       set_log_level, set_extern_logger, aclshmem_signal_wait_until)

from ._pyshmem import aclshmem_finalize as aclshmem_finialize  # aclshmem_finialize will be deprecated, use aclshmem_finalize instead

__all__ = [
    'aclshmem_init',
    'aclshmem_get_unique_id',
    'aclshmem_init_using_unique_id',
    'aclshmem_finalize',
    'aclshmem_malloc',
    'aclshmem_free',
    'aclshmem_ptr',
    'my_pe',
    'pe_count',
    'set_conf_store_tls_key',
    'set_conf_store_tls',
    'team_my_pe',
    'team_n_pes',
    'team_split_strided',
    'team_split_2d',
    'team_translate_pe',
    'team_destroy',
    'InitAttr',
    'InitStatus',
    'OpEngineType',
    'aclshmem_calloc',
    'aclshmem_align',
    'aclshmemx_init_status',
    'get_ffts_config',
    'aclshmem_global_exit',
    'aclshmem_putmem_nbi',
    'aclshmem_getmem_nbi',
    'aclshmemx_putmem_signal',
    'aclshmemx_putmem_signal_nbi',
    'aclshmem_putmem',
    'aclshmem_getmem',
    'aclshmem_info_get_version',
    'aclshmem_info_get_name',
    'aclshmem_team_get_config',
    'set_log_level',
    'set_extern_logger',
    'aclshmem_create_tensor',
    'aclshmem_free_tensor',
    'aclshmem_signal_wait_until'
]

def aclshmem_create_tensor(shape, dtype: torch.dtype=torch.float32, device_id=0) -> torch.Tensor:
    nbytes = calc_nbytes(shape, dtype)
    data_ptr = aclshmem_malloc(nbytes)

    if data_ptr == 0:
        raise RuntimeError("aclshmem_malloc failed")

    device = torch.device(f"npu:{device_id}")
    tensor = construct_tensor_from_ptr(data_ptr, shape, dtype, device)
    return tensor

def aclshmem_free_tensor(tensor: torch.Tensor):
    data_ptr = tensor.data_ptr()
    aclshmem_free(data_ptr)