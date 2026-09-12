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


def check_array(name, actual, golden):
    if actual.shape != golden.shape:
        raise AssertionError(f"{name} shape mismatch: {actual.shape} != {golden.shape}")
    if np.issubdtype(actual.dtype, np.integer):
        ok = np.array_equal(actual, golden)
    else:
        ok = np.allclose(actual.astype(np.float32), golden.astype(np.float32), rtol=1e-3, atol=1e-3)
    if not ok:
        idx = np.argwhere(actual != golden)
        first = tuple(idx[0]) if idx.size else 0
        raise AssertionError(f"{name} mismatch at {first}: actual={actual[first]}, golden={golden[first]}")


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
    max_recv_tokens = args.pes * args.bs * args.topk
    segment_num = args.expert_per_pe * args.pes
    case_dir = Path("golden") / f"shape_{args.bs}_{args.h}_{args.topk}_{moe_expert_num}_{args.pes}"
    output_dir = Path("output")

    for rank in range(args.pes):
        rank_dir = case_dir / f"rank_{rank}"
        check_array(
            f"rank {rank} expand_x",
            np.fromfile(output_dir / f"expand_x_{rank}.bin", dtype=dtype).reshape(max_recv_tokens, args.h),
            np.fromfile(rank_dir / "golden_expand_x.bin", dtype=dtype).reshape(max_recv_tokens, args.h),
        )
        check_array(
            f"rank {rank} assist_info",
            np.fromfile(output_dir / f"assist_info_{rank}.bin", dtype=np.int32).reshape(max_recv_tokens, 3),
            np.fromfile(rank_dir / "golden_assist_info.bin", dtype=np.int32).reshape(max_recv_tokens, 3),
        )
        check_array(
            f"rank {rank} ep_recv_count",
            np.fromfile(output_dir / f"ep_recv_count_{rank}.bin", dtype=np.int32).reshape(segment_num),
            np.fromfile(rank_dir / "golden_ep_recv_count.bin", dtype=np.int32).reshape(segment_num),
        )
        check_array(
            f"rank {rank} expert_token_nums",
            np.fromfile(output_dir / f"expert_token_nums_{rank}.bin", dtype=np.int32).reshape(args.expert_per_pe),
            np.fromfile(rank_dir / "golden_expert_token_nums.bin", dtype=np.int32).reshape(args.expert_per_pe),
        )

    print("[Dispatch] check passed")


if __name__ == "__main__":
    main()
