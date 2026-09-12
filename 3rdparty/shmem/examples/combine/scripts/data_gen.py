#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
import argparse
import json
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
            raise ImportError("bfloat16_t data generation requires ml_dtypes")
        return bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def gen_x(rng, shape, dtype):
    if dtype == np.int32:
        return rng.integers(-5, 6, size=shape, dtype=np.int32)
    return rng.uniform(-1.0, 1.0, size=shape).astype(dtype)


def dispatch_golden(x_all, expert_ids_all, pe_size, bs, h, topk, expert_per_pe):
    max_recv_tokens = pe_size * bs * topk
    segment_num = expert_per_pe * pe_size
    expand = []
    assist = []
    ep_recv_count = []

    for dst_rank in range(pe_size):
        rank_expand = np.zeros((max_recv_tokens, h), dtype=x_all.dtype)
        rank_assist = np.zeros((max_recv_tokens, 3), dtype=np.int32)
        running = 0
        rank_ep_recv = np.zeros((segment_num,), dtype=np.int32)

        for local_expert in range(expert_per_pe):
            expert_id = dst_rank * expert_per_pe + local_expert
            for src_rank in range(pe_size):
                count = 0
                for token_id in range(bs):
                    for topk_id in range(topk):
                        if int(expert_ids_all[src_rank, token_id, topk_id]) != expert_id:
                            continue
                        rank_expand[running + count] = x_all[src_rank, token_id]
                        rank_assist[running + count] = [src_rank, token_id, topk_id]
                        count += 1
                running += count
                rank_ep_recv[local_expert * pe_size + src_rank] = running

        expand.append(rank_expand)
        assist.append(rank_assist)
        ep_recv_count.append(rank_ep_recv)

    return expand, assist, ep_recv_count


def combine_golden(x_all, expert_ids_all, expert_scales_all):
    pe_size, bs, topk = expert_ids_all.shape
    h = x_all.shape[-1]
    out = np.zeros((pe_size, bs, h), dtype=np.float32)
    for rank in range(pe_size):
        for token_id in range(bs):
            for topk_id in range(topk):
                out[rank, token_id] += (
                    x_all[rank, token_id].astype(np.float32) * expert_scales_all[rank, token_id, topk_id]
                )
    return out.astype(x_all.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pes", type=int, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--expert-per-pe", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="int32_t")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dtype = dtype_from_name(args.dtype)
    moe_expert_num = args.pes * args.expert_per_pe
    rng = np.random.default_rng(args.seed)

    x_all = np.stack([gen_x(rng, (args.bs, args.h), dtype) for _ in range(args.pes)])
    expert_ids_all = rng.integers(0, moe_expert_num, size=(args.pes, args.bs, args.topk), dtype=np.int32)
    expert_scales_all = rng.uniform(0.05, 1.0, size=(args.pes, args.bs, args.topk)).astype(np.float32)
    if dtype == np.int32:
        expert_scales_all.fill(1.0)

    expand, assist, ep_recv_count = dispatch_golden(
        x_all, expert_ids_all, args.pes, args.bs, args.h, args.topk, args.expert_per_pe
    )
    x_out = combine_golden(x_all, expert_ids_all, expert_scales_all)

    out_dir = Path("golden") / f"shape_{args.bs}_{args.h}_{args.topk}_{moe_expert_num}_{args.pes}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "pe_size": args.pes,
        "bs": args.bs,
        "h": args.h,
        "topk": args.topk,
        "expert_per_pe": args.expert_per_pe,
        "moe_expert_num": moe_expert_num,
        "dtype": args.dtype,
        "expert_compute": "identity",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for rank in range(args.pes):
        rank_dir = out_dir / f"rank_{rank}"
        rank_dir.mkdir(exist_ok=True)
        expand[rank].tofile(rank_dir / "expand_x.bin")
        assist[rank].tofile(rank_dir / "assist_info.bin")
        ep_recv_count[rank].tofile(rank_dir / "ep_recv_count.bin")
        expert_ids_all[rank].tofile(rank_dir / "expert_ids.bin")
        expert_scales_all[rank].tofile(rank_dir / "expert_scales.bin")
        x_out[rank].tofile(rank_dir / "golden_x_out.bin")

    print(f"[Combine] generated data in {out_dir}")


if __name__ == "__main__":
    main()
