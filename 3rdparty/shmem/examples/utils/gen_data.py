# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import numpy as np
import torch
import torch_npu
from utils import CommType, DataType, tensor_to_file


def gen_random_data(size, dtype):
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32:
        return torch.randn(size=size, dtype=dtype)
    elif dtype == torch.int8:
        return torch.randint(-16, 16, size=size, dtype=dtype)
    else:
        print(f"Invalid dtype: {dtype}.")
        raise ValueError(f"Invalid dtype: {dtype}")


def gen_golden_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('comm_type', type=CommType.from_str,
                        choices=[CommType.MATMUL_ALLREDUCE,
                                 CommType.ALLGATHER_MATMUL,
                                 CommType.MATMUL_REDUCE_SCATTER,
                                 CommType.ALLGATHER_MATMUL_PADDING,
                                 CommType.MATMUL_REDUCE_SCATTER_PADDING,
                                 CommType.ALLGATHER_MATMUL_WITH_GATHER_RESULT])
    parser.add_argument('out_dtype', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16])
    parser.add_argument('pe_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('transA', type=int)
    parser.add_argument('transB', type=int)
    parser.add_argument('data_dir', type=str,
                        help='Directory to save the data files',
                        default="./out")
    args = parser.parse_args()
    m, n, k = args.m, args.n, args.k
    data_dir = os.path.abspath(args.data_dir)

    os.makedirs(data_dir, exist_ok=True)
    b_all_pe = gen_random_data([k, n], dtype=args.out_dtype.torch_type)

    matrix_a_list = []
    matrix_c_list_fp32 = []
    matrix_c_list_fp16_npu = []
    for i in range(args.pe_size):
        a_gm = gen_random_data([m, k], dtype=args.out_dtype.torch_type)
        matrix_a_list.append(a_gm)
        b_gm = b_all_pe
        
        a_np = a_gm.to(torch.float32).numpy()
        b_np = b_gm.to(torch.float32).numpy()
        matrix_c_fp32 = np.matmul(a_np, b_np)
        matrix_c_list_fp32.append(matrix_c_fp32)
        
        a_torch = a_gm.npu()
        b_torch = b_gm.npu()
        matrix_c_fp16_npu = torch.matmul(a_torch, b_torch)
        matrix_c_list_fp16_npu.append(matrix_c_fp16_npu)
        
        if args.transA:
            a_gm = a_gm.transpose(0, 1).contiguous()
        if args.transB:
            b_gm = b_gm.transpose(0, 1).contiguous()

        a_gm_path = os.path.join(data_dir, f"pe_{i}_a.bin")
        b_gm_path = os.path.join(data_dir, f"pe_{i}_b.bin")
        tensor_to_file(a_gm, a_gm_path)
        tensor_to_file(b_gm, b_gm_path)

    golden = None
    torch_output = None
    if (args.comm_type in
        [CommType.ALLGATHER_MATMUL, CommType.ALLGATHER_MATMUL_PADDING, CommType.ALLGATHER_MATMUL_WITH_GATHER_RESULT]):
        golden = np.concatenate(matrix_c_list_fp32, axis=0)
        torch_output = torch.cat([t.cpu() for t in matrix_c_list_fp16_npu], dim=0)
    else:
        golden = np.zeros_like(matrix_c_list_fp32[0])
        torch_output = torch.zeros_like(matrix_c_list_fp16_npu[0].cpu())
        for i in range(args.pe_size):
            golden += matrix_c_list_fp32[i]
            torch_output += matrix_c_list_fp16_npu[i].cpu()

    tensor_to_file(torch_output, os.path.join(data_dir, "torch_output.bin"))
    golden.tofile(os.path.join(data_dir, "golden.bin"))

    if args.comm_type == CommType.ALLGATHER_MATMUL_WITH_GATHER_RESULT:
        tensor_to_file(torch.cat(matrix_a_list, dim=0).to(torch.float32), os.path.join(data_dir, "gather_a.bin"))


if __name__ == '__main__':
    gen_golden_data()
