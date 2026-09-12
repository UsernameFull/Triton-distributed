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


def gen_random_data(size, dtype):
    if dtype == np.float16 or dtype == np.float32:
        return np.random.uniform(size=size).astype(dtype)
    else:
        print(f"Invalid dtype: {dtype}.")
        return None


def gen_golden_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        help='Directory to save the data files', 
                        default="./out")
    parser.add_argument('--out_data_type', type=int)
    parser.add_argument('--pe_size', type=int)
    parser.add_argument('--m', type=int)
    parser.add_argument('--n', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--transA', type=int)
    parser.add_argument('--transB', type=int)
    args = parser.parse_args()
    m, n, k = args.m, args.n, args.k
    data_dir = os.path.abspath(args.data_dir)

    os.makedirs(data_dir, exist_ok=True)
    out_data_type = np.float32 if args.out_data_type == 0 else np.float16
    l0c_dtype = np.float32

    golden = np.zeros((m, n), dtype=l0c_dtype)
    torch_output = torch.zeros((m, n), dtype=torch.float16).npu()
    np.random.seed(42)

    for i in range(args.pe_size):
        a_gm = gen_random_data((m, k), np.float16)
        b_gm = gen_random_data((k, n), np.float16)

        a_gm_path = os.path.join(data_dir, f"pe_{i}_a.bin")
        b_gm_path = os.path.join(data_dir, f"pe_{i}_b.bin")
        print(f'{a_gm_path=}')
        print(f'{b_gm_path=}')
        a_gm.tofile(a_gm_path)
        b_gm.tofile(b_gm_path)

        matrix_c = np.matmul(a_gm.astype(l0c_dtype), b_gm.astype(l0c_dtype))
        golden += matrix_c

        a_torch = torch.from_numpy(a_gm).npu()
        b_torch = torch.from_numpy(b_gm).npu()
        matrix_c_torch = torch.matmul(a_torch, b_torch)
        torch_output += matrix_c_torch

    cpu_output = torch_output.cpu().numpy()
    cpu_output_path = os.path.join(data_dir, "torch_output.bin")
    print(f'{cpu_output_path=}')
    cpu_output.tofile(cpu_output_path)

    golden_path = os.path.join(data_dir, "golden.bin")
    print(f'{golden_path=}')
    golden.tofile(golden_path)

if __name__ == '__main__':
    gen_golden_data()
