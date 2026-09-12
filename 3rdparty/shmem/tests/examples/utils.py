# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pytest


def get_rtol(dtype_str: str, computation_count):
    """
    根据数据类型和计算量动态调整误差阈值
    考虑到 MatMul+AllReduce 的累积误差特性，适当放宽阈值
    """
    if dtype_str == "fp16":
        err = 2**-8 if computation_count < 2048 else 2**-7
    elif dtype_str == "bf16":
        err = 2**-7 if computation_count < 2048 else 2**-6
    elif dtype_str == "fp32":
        if computation_count < 2048:
            err = 2**-11
        elif computation_count < 16384:
            err = 2**-10
        else:
            err = 2**-9
    else:
        # Should not happen with current config, but good practice to handle
        pytest.fail(f"Unsupported dtype for precision check: {dtype_str}")

    return err
