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
DTYPE = torch.float16
INPUT_SHAPE = (1024, 128)
# 初始占位，后续通过参数更新
PES = DEFAULT_PES
OUTPUT_SHAPE = (PES,) + INPUT_SHAPE


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
class AllGatherData:
    local_input: torch.Tensor
    golden_output: torch.Tensor

def gen_allgather_data(pe):
    torch.manual_seed(42 + pe)

    if DTYPE in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        local_input = torch.randn(INPUT_SHAPE, dtype=DTYPE)
    else:
        local_input = torch.randint(-128, 128, INPUT_SHAPE, dtype=DTYPE)
    golden_output = torch.zeros(OUTPUT_SHAPE, dtype=DTYPE)
    for r in range(PES):
        torch.manual_seed(42 + r)
        if DTYPE in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            golden_output[r] = torch.randn(INPUT_SHAPE, dtype=DTYPE)
        else:
            golden_output[r] = torch.randint(-128, 128, INPUT_SHAPE, dtype=DTYPE)
    return AllGatherData(local_input=local_input, golden_output=golden_output)


def worker(pe):
    load_torch_library('aclshmem_torch.so')
    aclshmem_common = torch.classes.ShmemOps.Manager()
    torch_npu.npu.set_device(pe)
    local_mem_size = 1024 * 1024 * 1024
    ipports = "tcp://127.0.0.1:8662"
    aclshmem_common.attr_init(pe, PES, local_mem_size, ipports)
    print(f"PE {pe} ACLShmem init success!")

    allgather = torch.classes.ShmemOps.AllGather()

    data = gen_allgather_data(pe)
    local_input_npu = data.local_input.npu(non_blocking=True)
    aclshmem_output = torch.zeros(OUTPUT_SHAPE, dtype=DTYPE).npu(device=f'npu:{torch_npu.npu.current_device()}')
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
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
                ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config) as prof:

            for _ in range(10):
                allgather.compute(aclshmem_output, local_input_npu)
                torch_npu.npu.synchronize()
                prof.step()
    elif TOOL == 0:
        allgather.compute(aclshmem_output, local_input_npu)
        torch_npu.npu.synchronize()
    else:
        print("Unknown tool, exiting...")
        return

    npu_output_cpu = aclshmem_output.cpu()

    is_correct = torch.equal(npu_output_cpu, data.golden_output)
    if not is_correct:
        print(f"[PE {pe}] golden_output:{data.golden_output.flatten()[:10]}")
        print(f"[PE {pe}] npu_output_cpu:{npu_output_cpu.flatten()[:10]}")
        raise AssertionError(f"[PE {pe}] AllGather result check failed!")

    print(f"[PE {pe}] Result check success, task completed!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allgather")

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
    OUTPUT_SHAPE = (PES,) + INPUT_SHAPE

    processes = []

    for pe in range(PES):
        p = multiprocessing.Process(target=worker, args=(pe,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes have finished")
