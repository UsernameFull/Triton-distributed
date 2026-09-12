#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
import argparse
from pathlib import Path

import numpy as np

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None


def dtype_from_name(name):
    if name in ("int", "int32_t"):
        return np.int32
    if name == "float16_t":
        return np.float16
    if name == "bfloat16_t":
        if bfloat16 is None:
            raise ImportError("bfloat16_t check requires ml_dtypes")
        return bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pes", type=int, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--expert-per-pe", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="int32_t")
    args = parser.parse_args()

    dtype = dtype_from_name(args.dtype)
    moe_expert_num = args.pes * args.expert_per_pe
    case_dir = Path("golden") / f"shape_{args.bs}_{args.h}_{args.topk}_{moe_expert_num}_{args.pes}"
    output_dir = Path("output")
    rtol = 0 if dtype == np.int32 else 1e-2
    atol = 0 if dtype == np.int32 else 1e-2

    for rank in range(args.pes):
        actual_path = output_dir / f"x_out_{rank}.bin"
        actual_flat = np.fromfile(actual_path, dtype=dtype)
        expected_size = args.bs * args.h
        if actual_flat.size != expected_size:
            raise ValueError(
                f"{actual_path} has {actual_flat.size} elements, expected {expected_size} "
                f"for shape ({args.bs}, {args.h})"
            )
        actual = actual_flat.reshape(args.bs, args.h)
        golden = np.fromfile(case_dir / f"rank_{rank}" / "golden_x_out.bin", dtype=dtype).reshape(args.bs, args.h)
        if dtype == np.int32:
            ok = np.array_equal(actual, golden)
        else:
            ok = np.allclose(actual.astype(np.float32), golden.astype(np.float32), rtol=rtol, atol=atol)
        if not ok:
            diff = np.abs(actual.astype(np.float32) - golden.astype(np.float32))
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            raise AssertionError(
                f"rank {rank} x_out mismatch at {idx}: actual={actual[idx]}, golden={golden[idx]}"
            )

    print("[Combine] check passed")


if __name__ == "__main__":
    main()
