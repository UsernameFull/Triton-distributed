# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import hashlib
import random
from functools import reduce
from contextlib import closing
import os
import subprocess
import socket
import multiprocessing
import numpy as np
import pytest

# import tests greneral configs.
from tests.examples.config import NP_RANDOM_SEED, SHAPE_TOTAL_SIZE_LIMIT
from tests.examples.np_normal_generator import NPNormalGenerator
from tests.examples.np_uniform_generator import NPUniformGenerator
from tests.examples.utils import get_rtol
from tests.examples.config import SHAPE_DIM_VALUES
from tests.examples.config import SHAPE_DIM_RANDOM_RANGE
from tests.examples.config import MM_AR_OP_CORRECTNESS_INPUT_RANGE
from tests.examples.config import NUMPY_DTYPES
from tests.examples.config import SUPPORT_RANKS
from tests.examples.config import CORRECTNESS_TEST_NUM_CASES_PER_DTYPE
from tests.examples.config import NUMERICAL_STABILITY_TEST_NUM_CASES_PER_DTYPE

# Use hardcoded paths as fixtures are not reliable
EXECUTABLE_PATH = os.path.abspath("build/bin/matmul_allreduce")
TEST_DATA_DIR = "tests/examples/matmul_allreduce/test_data"


def _product(factors):
    return reduce(lambda x, y: x * y, factors, 1)


def generate_shapes(num_cases=1):
    """Generates random tensor shapes for matmul based on constraints."""
    generated_shapes = set()
    # Limit combinations to avoid excessive generation time
    all_dim_values = SHAPE_DIM_VALUES[:10] + list(
        range(SHAPE_DIM_RANDOM_RANGE[0], SHAPE_DIM_RANDOM_RANGE[1], 64)
    )

    while len(generated_shapes) < num_cases:
        m = random.choice(all_dim_values)
        k = random.choice(all_dim_values)
        n = random.choice(all_dim_values)

        shape_a = (m, k)
        shape_b = (k, n)

        if (
                _product(shape_a) < SHAPE_TOTAL_SIZE_LIMIT
                and _product(shape_b) < SHAPE_TOTAL_SIZE_LIMIT
        ):
            generated_shapes.add((m, k, n))

    return [{"m": m, "k": k, "n": n} for m, k, n in generated_shapes]


def _generate_test_case(dtype_str, shape_info, world_size, category):
    """生成单个测试用例的通用逻辑"""
    m, k, n = shape_info["m"], shape_info["k"], shape_info["n"]
    id_str = f"mm-ar-{category}-test-{dtype_str}-w{world_size}-m{m}k{k}n{n}"
    return pytest.param({
        "world_size": world_size,
        "dtype": dtype_str,
        **shape_info,
        "category": category
    }, id=id_str)


def get_test_cases(
        num_cases_per_dtype_for_correctness=CORRECTNESS_TEST_NUM_CASES_PER_DTYPE,
        num_cases_per_dtype_for_stability=NUMERICAL_STABILITY_TEST_NUM_CASES_PER_DTYPE,
):
    """Generates a list of test cases."""
    test_cases = []
    for dtype_str in ["fp16"]:  # 可扩展为 ["fp16", "fp32", "bf16"]
        # 处理正确性测试用例
        for shape_info in generate_shapes(num_cases_per_dtype_for_correctness):
            for world_size in SUPPORT_RANKS:
                test_cases.append(_generate_test_case(
                    dtype_str, shape_info, world_size, "correctness"
                ))

        # 处理稳定性测试用例
        for shape_info in generate_shapes(num_cases_per_dtype_for_stability):
            for world_size in SUPPORT_RANKS:
                test_cases.append(_generate_test_case(
                    dtype_str, shape_info, world_size, "stability"
                ))
    return test_cases


# Test implementation
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def run_fusion_matmul_allreduce_kernel(
        rank, case_params, ipport, base_device_id, executable_path, test_data_dir
):
    """The function to be executed by each rank's process."""
    world_size = case_params["world_size"]
    m, k, n = case_params["m"], case_params["k"], case_params["n"]

    # Launch the C++ executable
    cmd = [
        executable_path,
        str(world_size),
        str(rank),
        ipport,
        str(base_device_id),
        str(m),
        str(k),
        str(n),
        test_data_dir,
    ]

    # It's better to capture stdout/stderr for debugging
    log_path = os.path.join(test_data_dir, "log.txt")
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, cwd=test_data_dir, stdout=log_file, stderr=subprocess.STDOUT
        )
        proc.wait()

    if proc.returncode != 0:
        # This allows pytest to show the logs on failure
        with open(log_path, "r") as f:
            print(f"--- RANK {rank} LOGS ---")
            print(f.read())
        pytest.fail(
            f"Rank {rank} failed with exit code {proc.returncode}", pytrace=False
        )


@pytest.mark.parametrize("case_params", get_test_cases())
def test_fusion_matmul_allreduce(case_params):
    """Main test function for matmul_allreduce kernel."""
    if not os.path.exists(EXECUTABLE_PATH):
        pytest.skip(f"Executable not found at {EXECUTABLE_PATH}, run build.sh first.")

    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    world_size = case_params["world_size"]
    m, k, n = case_params["m"], case_params["k"], case_params["n"]
    dtype_str = case_params["dtype"]
    numpy_dtype = NUMPY_DTYPES.get(dtype_str, np.float32)

    # Setup networking
    master_port = find_free_port()
    master_addr = "127.0.0.1"
    ipport = f"tcp://{master_addr}:{master_port}"
    base_device_id = 0

    shape_a = (m, k)
    shape_b = (k, n)
    shape_c = (m, n)

    numpy_dtype = NUMPY_DTYPES.get(dtype_str, np.float16)
    # For reproducibility, let's re-seed before data generation
    # op standard data generation
    np.random.seed(NP_RANDOM_SEED)
    case_category = case_params["category"]
    if "correctness" in case_category:
        in_low, in_hi = MM_AR_OP_CORRECTNESS_INPUT_RANGE
        np_data_generator = NPUniformGenerator(
            low=in_low, hi=in_hi, output_dtype=numpy_dtype
        )

    elif "stability" in case_category:
        np_data_generator = NPNormalGenerator(output_dtype=numpy_dtype)

    # all ranks input same tensor.
    all_a = [np_data_generator.generate(shape_a)] * world_size
    all_b = [np_data_generator.generate(shape_b)] * world_size

    # cal CPU matmul & allreduce.
    gt_fp32 = np.zeros(shape_c, dtype=np.float32)
    case_id_str = f"{dtype_str}-w{world_size}-m{m}k{k}n{n}"

    for i in range(world_size):
        # Always calculate matmul in fp32 for precision
        a_i = all_a[i]
        b_i = all_b[i]

        mm = np.matmul(a_i.astype(np.float32), b_i.astype(np.float32))
        # Skip if intermediate matmul overflows, as the test case is not meaningful
        if np.isposinf(mm).any() or np.isneginf(mm).any() or np.isnan(mm).any():
            print(
                f"\nINFO: Overflow in intermediate matmul for rank {i} in case {case_id_str}. Skipping."
            )
            pytest.skip("Skipping test due to overflow in intermediate matmul.")

        gt_fp32 += mm

    gt = gt_fp32.astype(numpy_dtype).reshape(-1)

    case_hash = hashlib.md5(str(case_params).encode()).hexdigest()
    case_params["case_id"] = case_hash
    # data_dir is independent for every single test case.
    data_dir = os.path.abspath(os.path.join(TEST_DATA_DIR, case_hash))
    os.makedirs(data_dir, exist_ok=True)
    for i in range(world_size):
        rank_i_a_path = os.path.abspath(os.path.join(data_dir, f"rank_{i}_a.bin"))
        rank_i_b_path = os.path.abspath(os.path.join(data_dir, f"rank_{i}_b.bin"))
        with open(rank_i_a_path, "wb") as f:
            f.write(all_a[i].astype(numpy_dtype).tobytes())

        with open(rank_i_b_path, "wb") as f:
            f.write(all_b[i].astype(numpy_dtype).tobytes())

    # pack CPU input & output.
    case_params[case_hash] = {"A": all_a[i], "B": all_b[i], "gt": gt}

    # Run ranks in parallel
    ctx = multiprocessing.get_context("spawn")
    processes = []
    for rank_id in range(world_size):
        p = ctx.Process(
            target=run_fusion_matmul_allreduce_kernel,
            args=(
                rank_id,
                case_params,
                ipport,
                base_device_id,
                EXECUTABLE_PATH,
                data_dir,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        assert p.exitcode == 0

    aclshmem_output_path = os.path.join(data_dir, "aclshmem_output.bin")
    aclshmem_result_data = np.fromfile(aclshmem_output_path, dtype=numpy_dtype)
    act = aclshmem_result_data.reshape(-1)

    # 计算次数公式：每个rank做矩阵乘法 + AllReduce累加
    # MatMul: world_size * m * k * n (每个rank的矩阵乘法)
    # AllReduce: m * n * (world_size - 1) (累加操作)
    cmp_count = world_size * m * k * n + m * n * (world_size - 1)
    err = get_rtol(dtype_str, cmp_count)
    # Mask for relative error check: |golden| >= 1.0
    rel_err_check_mask = np.abs(gt) >= 1.0
    if rel_err_check_mask.any():
        re = np.abs(act[rel_err_check_mask] - gt[rel_err_check_mask]) / (
                np.abs(gt[rel_err_check_mask]) + 1e-7
        )
        max_re = re.max().item()
        assert max_re <= err, f"Relative error check failed for {aclshmem_output_path}!"
        "Max RE = {max_re:.4e} > threshold ({err:.4e})"

    # Mask for absolute error check: |golden| < 1.0
    abs_err_check_mask = np.abs(gt) < 1.0
    if abs_err_check_mask.any():
        ae = np.abs(act[abs_err_check_mask] - gt[abs_err_check_mask])
        max_ae = ae.max().item()
        assert max_ae <= err, f"Absolute error check failed for {aclshmem_output_path}! "
        "Max AE = {max_ae:.4e} > threshold ({err:.4e})"
