# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import sys
from enum import Enum
import traceback

import numpy as np
import scipy
import torch

from utils import DataType, tensor_from_file, get_rtol

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


class OpTypes(Enum):
    NA = 0 # new standard is not available
    MOVE = 1
    RAND = 2
    CAST = 3
    COMPUTE_INTEGER = 4
    COMPUTE_QUANT = 5
    COMPUTE_FLOAT = 6
    COMPUTE_FLOAT_HIGH_PRECISION = 7
    VECTOR_FUSION = 8
    CV_FUSION = 9


def get_precision_and_eb_threshold(op_type, dtype, rtol: float = 2**(-8)):
    precision_threshold = 0
    eb_threshold = 0
    if op_type in [OpTypes.MOVE, OpTypes.RAND, OpTypes.CAST, OpTypes.COMPUTE_INTEGER]:
        pass
    if op_type in [OpTypes.COMPUTE_QUANT]:
        if dtype in [torch.int8]:
            precision_threshold = 1
    if op_type in [OpTypes.COMPUTE_QUANT, OpTypes.COMPUTE_FLOAT]:
        if dtype in [torch.float16]:
            precision_threshold = rtol
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            precision_threshold = rtol
            eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            precision_threshold = rtol
            eb_threshold = 2**(-14)
    if op_type in [OpTypes.COMPUTE_FLOAT_HIGH_PRECISION]:
        if dtype in [torch.float16]:
            precision_threshold = 2**(-11)
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            precision_threshold = 2**(-8)
            eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            precision_threshold = 2**(-14)
            eb_threshold = 2**(-14)
    if op_type in [OpTypes.VECTOR_FUSION]:
        if dtype in [torch.float16]:
            precision_threshold = 2**(-9)
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            precision_threshold = 2**(-8)
            eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            precision_threshold = 2**(-12)
            eb_threshold = 2**(-14)
    if op_type in [OpTypes.CV_FUSION]:
        precision_threshold = 522 # 最大相对误差5/平均相对误差2/均方根误差2
        if dtype in [torch.float16]:
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            eb_threshold = 2**(-14)
    logging.debug(
        "op_type: %s, dtype: %s, precision_threshold: %s, eb_threshold: %s",
        op_type,
        dtype,
        precision_threshold,
        eb_threshold
    )
    return precision_threshold, eb_threshold


def precision_performance_analysis(op_type, golden_output_tensor_list, output_tensor_list, rtol: float):
    for i, golden_output in enumerate(golden_output_tensor_list):
        actual_output = output_tensor_list[i].cpu()
        precision_threshold, eb_threshold = get_precision_and_eb_threshold(op_type, actual_output.dtype, rtol)
        precision, eb = cal_precision_eb_percent(op_type, actual_output, golden_output,
                                                 precision_threshold, eb_threshold)
    print(f"Old precision - precision: {precision}, eb: {eb}", flush=True)
    if precision == 100 and eb <= 100:
        return True
    else:
        return False


def cal_precision_eb_percent(op_type, actual_output, golden_output, precision_threshold, eb_threshold):
    actual_output = actual_output if actual_output.dtype != torch.bool else actual_output.long()
    golden_output = golden_output if golden_output.dtype != torch.bool else golden_output.long()
    if (op_type in [OpTypes.COMPUTE_FLOAT, OpTypes.COMPUTE_FLOAT_HIGH_PRECISION, OpTypes.VECTOR_FUSION] and
        actual_output.dtype in [torch.float16, torch.bfloat16]):
        actual_output = actual_output.to(torch.float32)
        golden_output = golden_output.to(torch.float32)
    actual_output = torch.where(torch.isnan(actual_output), torch.full_like(actual_output, 0), actual_output)
    actual_output = torch.where(torch.isinf(actual_output), torch.full_like(actual_output, 0), actual_output)
    golden_output = torch.where(torch.isnan(golden_output), torch.full_like(golden_output, 0), golden_output)
    golden_output = torch.where(torch.isinf(golden_output), torch.full_like(golden_output, 0), golden_output)
    if op_type == OpTypes.RAND:
        alpha = 0.01
        t_statistic, p_value = scipy.stats.ks_2samp(actual_output, golden_output)
        precision_percent = 100 if p_value > alpha else 0
        eb_percent = 0
        return precision_percent, eb_percent
    diff = torch.subtract(actual_output, golden_output)
    tensor_max = torch.maximum(torch.ones(golden_output.shape, dtype=golden_output.dtype), torch.abs(golden_output))
    if precision_threshold == 1:
        tolerance = torch.subtract(torch.abs(diff), torch.ones(diff.shape, dtype=diff.dtype))
    else:
        tolerance = torch.subtract(torch.abs(diff), precision_threshold * tensor_max)

    different_element_indexes = torch.where(tolerance > 0)[0]
    for index, real_index in enumerate(different_element_indexes):
        golden_data = golden_output[real_index]
        output_data = actual_output[real_index]
        print(
            f"data index: {real_index:6d}, expected: {golden_data:-.9f}, "
            f"actual: {output_data:-.9f}, rdiff: {abs(output_data - golden_data) / golden_data:-.6f}"
        )
        if index == 10:
            break
    error_num = len(different_element_indexes)
    if error_num > 0:
        print(f"Differential num: {error_num}")

    eb = eb_threshold
    if eb_threshold != 0:
        eb = torch.abs(torch.mean(torch.div(diff, tensor_max)))
    precision_percent = torch.sum(tolerance <= 0).numpy() / torch.numel(tolerance) * 100
    eb_percent = 0 if eb == 0 else torch.sum(eb).to(torch.float).numpy() / eb_threshold * 100
    return precision_percent, eb_percent


def calculate_metrics(actual, golden):
    actual = actual.to(torch.float32)
    golden = golden.to(torch.float32)
    
    actual = torch.where(torch.isnan(actual), torch.full_like(actual, 0), actual)
    actual = torch.where(torch.isinf(actual), torch.full_like(actual, 0), actual)
    golden = torch.where(torch.isnan(golden), torch.full_like(golden, 0), golden)
    golden = torch.where(torch.isinf(golden), torch.full_like(golden, 0), golden)
    
    diff = torch.abs(actual - golden)
    golden_abs = torch.abs(golden)
    denominator = golden_abs + 1e-7
    
    relative_error = diff / denominator
    mare = torch.max(relative_error).item()
    mere = torch.mean(relative_error).item()
    
    squared_error = diff ** 2
    mean_squared_error = torch.mean(squared_error).item()
    rmse = np.sqrt(mean_squared_error)
    
    return mare, mere, rmse


def verify_result():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str)
    parser.add_argument('golden', type=str)
    parser.add_argument('out_dtype', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16])
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('torch_output', nargs='?', default=None, type=str,
                        help='Torch_NPU output file for new precision check (optional)')
    args = parser.parse_args()

    output = tensor_from_file(args.output, dtype=args.out_dtype.torch_type)
    golden = tensor_from_file(args.golden, dtype=torch.float32)

    old_pass = False
    new_pass = False

    rtol = get_rtol(dtype=args.out_dtype.torch_type, compute_times=args.k)
    op_type = OpTypes.COMPUTE_FLOAT
    print(f"Running old precision check...")
    old_pass = precision_performance_analysis(op_type, [golden], [output], rtol)

    if args.torch_output is not None:
        print(f"Loading torch_output from: {args.torch_output}", flush=True)
        torch_output = tensor_from_file(args.torch_output, dtype=args.out_dtype.torch_type)
        print(f"torch_output loaded, shape: {torch_output.shape}", flush=True)
        print(f"Running new precision check...", flush=True)
        shmem_mare, shmem_mere, shmem_rmse = calculate_metrics(output, golden)
        torch_mare, torch_mere, torch_rmse = calculate_metrics(torch_output, golden)

        print(f"SHMEM metrics - MARE: {shmem_mare:.9f}, MERE: {shmem_mere:.9f}, RMSE: {shmem_rmse:.9f}", flush=True)
        print(f"Torch_NPU metrics - MARE: {torch_mare:.9f}, MERE: {torch_mere:.9f}, RMSE: {torch_rmse:.9f}", flush=True)

        mare_ratio = shmem_mare / torch_mare if torch_mare != 0 else float('inf')
        mere_ratio = shmem_mere / torch_mere if torch_mere != 0 else float('inf')
        rmse_ratio = shmem_rmse / torch_rmse if torch_rmse != 0 else float('inf')

        print(f"Ratios - MARE: {mare_ratio:.9f}, MERE: {mere_ratio:.9f}, RMSE: {rmse_ratio:.9f}", flush=True)

        mare_pass = mare_ratio <= 10
        mere_pass = mere_ratio <= 2
        rmse_pass = rmse_ratio <= 2

        new_pass = mare_pass and mere_pass and rmse_pass

        if not mare_pass:
            print(f"{RED}MARE ratio {mare_ratio:.9f} > 10{RESET}", flush=True)
        if not mere_pass:
            print(f"{RED}MERE ratio {mere_ratio:.9f} > 2{RESET}", flush=True)
        if not rmse_pass:
            print(f"{RED}RMSE ratio {rmse_ratio:.9f} > 2{RESET}", flush=True)

    return old_pass or new_pass


if __name__ == '__main__':
    try:
        res = verify_result()
        if not res:
            print(f"{RED}||||||||||| PRECISION ERROR |||||||||||||||{RESET}")
            sys.exit(1)
        else:
            print(f"{GREEN}PRECISION PASS{RESET}")
            sys.exit(0)
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(1)
