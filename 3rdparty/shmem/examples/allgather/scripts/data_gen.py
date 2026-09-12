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

from ml_dtypes import bfloat16

# Set seed for multi-node situation
np.random.seed(42)

def gen_random_data(size, dtype):
    return np.random.uniform(low=0.0, high=10.0, size=size).astype(dtype)


def golden_generate(data_len, pe_size, data_type):
    golden_dir = f"allgather_{data_len}_{pe_size}"
    cmd = f"mkdir golden/{golden_dir}"
    os.system(cmd)

    input_gm = np.zeros((pe_size, data_len), dtype=data_type)
    output_gm = np.zeros((pe_size * data_len), dtype=data_type)

    for i in range(pe_size):
        input_gm[i][:] = gen_random_data((data_len), dtype=data_type)
        output_gm[i * data_len: i * data_len + data_len] = input_gm[i]

    for i in range(pe_size):
        input_gm[i].tofile(f"./golden/{golden_dir}/input_gm_{i}.bin")
    output_gm.tofile(f"./golden/{golden_dir}/golden.bin")
    print(f"{data_len} golden generate success !")


def gen_golden_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pe_size', type=int)
    parser.add_argument('test_type', type=str)
    args = parser.parse_args()

    type_map = {
        "int": np.int32,
        "int32_t": np.int32,
        "float16_t": np.float16,
        "bfloat16_t": bfloat16
    }

    data_type = type_map.get(args.test_type, 'float16_t')
    pe_size = args.pe_size

    case_num = 24
    for i in range(case_num):
        data_len = 16 * (2 ** i)
        golden_generate(data_len, pe_size, data_type)


if __name__ == '__main__':
    gen_golden_data()
