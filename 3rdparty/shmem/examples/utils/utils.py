# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from enum import IntEnum
import numpy as np
import torch


class CommType(IntEnum):
    MATMUL_ALLREDUCE = 0
    ALLGATHER_MATMUL = 1
    MATMUL_REDUCE_SCATTER = 2
    MATMUL_REDUCE_SCATTER_PADDING = 3
    ALLGATHER_MATMUL_WITH_GATHER_RESULT = 4
    ALLGATHER_MATMUL_PADDING = 5

    @classmethod
    def from_str(cls, arg: str):
        return cls(int(arg))


class DataType(IntEnum):
    FLOAT = 0
    FLOAT16 = 1
    BF16 = 27

    @property
    def torch_type(self):
        return {
            DataType.FLOAT: torch.float,
            DataType.FLOAT16: torch.float16,
            DataType.BF16: torch.bfloat16,
        }[self]
    
    @classmethod
    def from_str(cls, arg: str):
        return cls(int(arg))


def tensor_to_file(tensor: torch.Tensor, file_name: str) -> None:
    if tensor.dtype == torch.bfloat16:
        tensor.to(torch.float32).numpy().tofile(file_name)
    else:
        tensor.numpy().tofile(file_name)


def tensor_from_file(file_name: str, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.bfloat16:
        return torch.from_numpy(np.fromfile(file_name, dtype=np.float32)).to(torch.bfloat16)
    else:
        numpy_dtype = torch.empty(0, dtype=dtype).numpy().dtype
        return torch.from_numpy(np.fromfile(file_name, numpy_dtype))


def get_rtol(dtype: torch.dtype, compute_times: int) -> float:
    if dtype == torch.float16:
        return 2 ** (-8) if compute_times < 2048 else 2 ** (-7)
    elif dtype == torch.bfloat16:
        return 2 ** (-7) if compute_times < 2048 else 2 ** (-6)
    elif dtype == torch.float32:
        return 2 ** (-11) if compute_times < 2048 else 2 ** (-10)
    else:
        raise ValueError(f"Invalid dtype: {dtype}.")