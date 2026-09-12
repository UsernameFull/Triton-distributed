# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
from glob import glob
import multiprocessing
from dataclasses import dataclass
import argparse
import torch
import torch_npu
import torch_npu.profiler

DEFAULT_PES = 8 
DTYPE = torch.float16  # 仅支持FP16
M = 1024
K = 128
N = 64
# 初始占位，后续通过参数更新
PES = DEFAULT_PES
A_SHAPE = (M, K)               # 单pe输入A (1024, 128)
B_SHAPE = (K, N)               # 权重B (128, 64)
C_SHAPE = (PES * M, N)

def load_torch_library(lib_name):
    lib_env = "LD_LIBRARY_PATH" if os.name == "posix" else "PATH"
    lib_paths = os.environ.get(lib_env, "").split(os.pathsep)
    for path in lib_paths:
        if not os.path.isdir(path):
            continue
        match = glob(os.path.join(path, lib_name))
        if match:
            lib_path = match[0]
            torch.ops.load_library(lib_path)
            return
    raise FileNotFoundError(
        f"Library {lib_name} not found in {lib_env} paths, please check the environment variable or library path"
    )

def tensor_size_in_bytes(tensor):
    num_elements = tensor.numel()
    element_size = tensor.untyped_storage().element_size()
    return num_elements * element_size

@dataclass
class AllGatherMatmulData:
    tensor_a: torch.Tensor       # 单pe输入A (1024, 128)
    tensor_b: torch.Tensor       # 权重B (128, 64)
    golden_c: torch.Tensor       # 输出C的golden

def gen_allgather_matmul_golden(pe):
    torch.manual_seed(100)
    tensor_b = torch.randn(B_SHAPE, dtype=DTYPE)
    
    torch.manual_seed(42 + pe)
    tensor_a = torch.randn(A_SHAPE, dtype=DTYPE)
    
    all_pes_matmul = []
    for r in range(PES):
        torch.manual_seed(42 + r)
        pe_a = torch.randn(A_SHAPE, dtype=DTYPE)
        pe_matmul = torch.matmul(pe_a.to(torch.float32), tensor_b.to(torch.float32))  # (1024, 64)
        all_pes_matmul.append(pe_matmul)
    
    golden_c = torch.cat(all_pes_matmul, dim=0)  # (8192, 64)
    
    return AllGatherMatmulData(
        tensor_a=tensor_a,
        tensor_b=tensor_b,
        golden_c=golden_c
    )

def worker(pe):
    load_torch_library('aclshmem_torch.so')
    
    aclshmem_common = torch.classes.ShmemOps.Manager()
    torch_npu.npu.set_device(pe)
    local_mem_size = 1024 * 1024 * 1024
    ipports = "tcp://127.0.0.1:8662"
    aclshmem_common.attr_init(pe, PES, local_mem_size, ipports)
    print(f"PE {pe} ACLShmem init success!")

    allgather_matmul = torch.classes.ShmemOps.AllGatherMatmul()

    data = gen_allgather_matmul_golden(pe)
    tensor_a_npu = data.tensor_a.npu(non_blocking=True)
    tensor_b_npu = data.tensor_b.npu(non_blocking=True)
    
    tensor_c_npu = torch.zeros(C_SHAPE, dtype=DTYPE).npu(device=f'npu:{torch_npu.npu.current_device()}')
    torch_npu.npu.synchronize()

    if TOOL == 1:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False
        )

        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./allgather_matmul_result"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config
        ) as prof:

            for _ in range(10):
                allgather_matmul.compute(tensor_c_npu, tensor_a_npu, tensor_b_npu)
                torch_npu.npu.synchronize()
                prof.step()
    elif TOOL == 0:
        allgather_matmul.compute(tensor_c_npu, tensor_a_npu, tensor_b_npu)
        torch_npu.npu.synchronize()
    else:
        print("Unknown tool, exiting...")
        return

    print(f"PE {pe} start ALLGATHER_MATMUL golden check...")
    tensor_c_cpu = tensor_c_npu.cpu()
    
    is_correct = torch.allclose(tensor_c_cpu.to(torch.float32), data.golden_c.to(torch.float32), rtol=1e-2, atol=1e-2)
    
    if not is_correct:
        print(f"[PE {pe}] golden_c sample:{data.golden_c.flatten()[:10]}")
        print(f"[PE {pe}] tensor_c_cpu sample:{tensor_c_cpu.flatten()[:10]}")
        print(f"[PE {pe}] golden shape: {data.golden_c.shape}, output shape: {tensor_c_cpu.shape}")
        raise AssertionError(f"[PE {pe}] ALLGATHER_MATMUL result check failed!")

    print(f"[PE {pe}] ALLGATHER_MATMUL result check success!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AllgatherMatmul")

    parser.add_argument(
        "--tool",
        type=int,
        default=0
    )
    parser.add_argument(
        "--pes",
        type=int,
        default=8
    )

    args = parser.parse_args()
    
    TOOL = args.tool
    PES = args.pes
    C_SHAPE = (PES * M, N)

    processes = []

    for pe in range(PES):
        p = multiprocessing.Process(target=worker, args=(pe,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes have finished")
