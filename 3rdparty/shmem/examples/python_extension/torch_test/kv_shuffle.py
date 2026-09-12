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
import argparse
from dataclasses import dataclass
import torch
import torch_npu
import torch_npu.profiler

PES = 8
TOOL = 0
# Model Params
MAX_SEQLEN = 1024
MAX_BATCH = 10
INIT_BATCH = 5

# KVCache Params
PAGE_SIZE = 128
max_block_nums = MAX_SEQLEN * MAX_BATCH // PAGE_SIZE
KV_HEAD_NUM = 8
HEAD_DIM = 128


def get_pair_pe(sort_idx, local_pe):
    pair_pe = 0
    for idx, pe in enumerate(sort_idx):
        if pe == local_pe:
            pair_idx = (PES - 1) - idx # notice idx can't be PES
            pair_pe = sort_idx[pair_idx]
    return pair_pe.item()


max_transfer_tokens = 16384 * 2


def get_pair_transfer_tokens(kv_lens, kv_sum, kv_mean, pair_list, local_pe):
    pair_pe = pair_list[local_pe][0]
    if (kv_sum[local_pe] - kv_mean) <= 0:
        return -1, []
    if (kv_sum[local_pe] - kv_mean) >= 0 and (kv_sum[pair_pe] - kv_mean) >= 0:
        return -1, []
    transfer_tokens = min(abs(kv_sum[local_pe] - kv_mean), abs(kv_sum[pair_pe] - kv_mean))
    transfer_tokens = min(transfer_tokens, max_transfer_tokens)
    transfer_batch_id = []
    transfer_batch_tokens = 0
    for i in range(kv_lens[local_pe].shape[0]): # pe_loop
        if transfer_tokens > kv_lens[local_pe][i]: # 选择哪几个batch的block要搬到pair_pe
            transfer_batch_id.append(i)
            transfer_tokens -= kv_lens[local_pe][i]
            transfer_batch_tokens += kv_lens[local_pe][i]
    return transfer_batch_tokens, transfer_batch_id


def balance_kv(kv_lens):
    kv_sum = torch.sum(kv_lens, dim=-1) # (pe_size, 1)
    kv_mean = torch.mean(kv_sum.float(), dim=0).int() # int
    sort_kv_sum, sort_idx = torch.sort(kv_sum)

    # Get pe to pe pair
    pair_list = []
    for i in range(PES):
        pair_idx = get_pair_pe(sort_idx, i)
        pair_list.append([pair_idx])

    # Get pe to pe transfer_tokens
    transfer_tokens_list = []
    for i in range(PES):
        transfer_tokens, transfer_batch_id = get_pair_transfer_tokens(kv_lens, kv_sum, kv_mean, pair_list, i)
        if (transfer_tokens > 0):
            pair_list[i].append(0) # 0 means send
        else:
            pair_list[i].append(1) # 1 means recv
        transfer_tokens_list.append((transfer_tokens, transfer_batch_id))
    return pair_list, transfer_tokens_list


@dataclass
class CacheData:
    pair_list: list
    k_cache_list: list
    v_cache_list: list
    src_block_table: list
    dst_block_table: list
    block_num_list: list
    k_cache_list_g: list
    v_cache_list_g: list


def gendata(pe):
    torch.manual_seed(42)
    kv_lens = torch.randint(0, MAX_SEQLEN, (PES, INIT_BATCH))
    kv_sum = torch.sum(kv_lens, dim=-1) # (pe_size, 1)

    k_cache_list = []
    v_cache_list = []
    used_blocks_list = []
    batch_blocks_list = []
    # Prepare Inputs
    for i in range(PES):
        k_cache = torch.zeros((max_block_nums, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM), dtype=torch.int8)
        v_cache = torch.zeros((max_block_nums, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM), dtype=torch.int8)
        batch_list = kv_lens[i]
        used_id = 0
        batch_blocks = []
        for j in range(batch_list.shape[0]):
            seqlen = batch_list[j].item()
            block_num = seqlen // PAGE_SIZE + 1
            k_cache_real = torch.randint(low=-128, high=128, size=(block_num, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM),
                                         dtype=torch.int8)
            v_cache_real = torch.randint(low=-128, high=128, size=(block_num, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM),
                                         dtype=torch.int8)
            k_cache[used_id:used_id + block_num] = k_cache_real
            v_cache[used_id:used_id + block_num] = v_cache_real
            batch_blocks.append((j, [_ for _ in range(used_id, used_id + block_num)]))
            used_id += block_num
        used_blocks_list.append(used_id)
        batch_blocks_list.append(batch_blocks)
        k_cache_list.append(k_cache)
        v_cache_list.append(v_cache)

    # Shuffle Calculate
    pair_list, transfer_tokens_list = balance_kv(kv_lens)

    # Record Input KV Cache

    src_block_table = []
    dst_block_table = []
    block_num_list = []
    k_cache_list_g = [_k_cache.clone() for _k_cache in k_cache_list]
    v_cache_list_g = [_v_cache.clone() for _v_cache in v_cache_list]

    # Params Prepare And Golden Calculate
    for i in range(PES):
        local_pe = i
        shuffle_table = pair_list
        k_cache_list_g = k_cache_list_g
        v_cache_list_g = v_cache_list_g
        src_local_table = []
        dst_local_table = []

        transfer_batches = transfer_tokens_list[local_pe][1]
        for batch_idx in transfer_batches:
            src_local_table += batch_blocks_list[local_pe][batch_idx][1]

        pair_pe = shuffle_table[local_pe][0]
        for new_block_id in range(len(src_local_table)):
            dst_used_id = used_blocks_list[pair_pe]
            dst_local_table.append(dst_used_id + new_block_id)

        src_block_table.append(src_local_table)
        dst_block_table.append(dst_local_table)
        block_num_list.append(len(src_local_table))

        # Start KVCache Copy
        total_data_volume = 0
        if (shuffle_table[local_pe][1] == 0):
            for idx, _ in enumerate(src_local_table):
                src_idx = src_local_table[idx]
                dst_idx = dst_local_table[idx]
                k_cache_list_g[pair_pe][dst_idx] = k_cache_list_g[local_pe][src_idx]
                v_cache_list_g[pair_pe][dst_idx] = v_cache_list_g[local_pe][src_idx]

                def tensor_size_in_bytes(tensor):
                    num_elements = tensor.numel()
                    element_size = tensor.untyped_storage().element_size()
                    return num_elements * element_size

                data_volume = tensor_size_in_bytes(k_cache_list_g[local_pe][src_idx])
                total_data_volume += data_volume
                data_volume = tensor_size_in_bytes(v_cache_list_g[local_pe][src_idx])
                total_data_volume += data_volume
            if pe == 0:
                print(f"pe:{local_pe}, datasize{total_data_volume}!")

    return CacheData(pair_list, k_cache_list, v_cache_list, src_block_table,
                     dst_block_table, block_num_list, k_cache_list_g, v_cache_list_g)


def read_file(file_name):
    with open(file_name, 'rb') as file:
        file_content = file.read()

    # 获取文件内容的字节数
    num_bytes = len(file_content)
    return file_content, num_bytes

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
        f"未在{lib_env}的路径中找到库{lib_name}，请检查环境变量或库路径"
    )

def worker(pe):
    # 加载共享库
    load_torch_library('aclshmem_torch.so')
    aclshmem_common = torch.classes.ShmemOps.Manager()
    torch_npu.npu.set_device(pe)
    local_mem_size = 1024 * 1024 * 1024
    ipports = "tcp://127.0.0.1:8662"
    aclshmem_common.attr_init(pe, PES, local_mem_size, ipports)
    kv_shuffle = torch.classes.ShmemOps.KVShuffle()
    my_cache_data = gendata(pe)
    
    # global_shuffle_table
    global_shuffle_table = torch.tensor(my_cache_data.pair_list, dtype=torch.int64)
    global_shuffle_tensor = global_shuffle_table.npu()
    # k_cache
    k_cache = my_cache_data.k_cache_list[pe]
    aclshmem_k_cache_tensor = aclshmem_common.malloc_like(k_cache)
    # v_cache
    v_cache = my_cache_data.v_cache_list[pe]
    aclshmem_v_cache_tensor = aclshmem_common.malloc_like(v_cache)
    int64_data = my_cache_data.block_num_list[pe]
    
    # 检查第一个值是否为 0
    if int64_data != 0:
        # src_block_table
        src_block_table = torch.tensor(my_cache_data.src_block_table[pe], dtype=torch.int64)
        src_block_tensor = src_block_table.npu()

        # dst_block_table
        dst_block_table = torch.tensor(my_cache_data.dst_block_table[pe], dtype=torch.int64)
        dst_block_tensor = dst_block_table.npu()
    else:
        src_block_tensor = torch.Tensor()
        dst_block_tensor = torch.Tensor()
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
                kv_shuffle.compute(global_shuffle_tensor, aclshmem_k_cache_tensor,
                                aclshmem_v_cache_tensor, src_block_tensor, dst_block_tensor)
                torch_npu.npu.synchronize() 
                prof.step()
    elif TOOL == 0:
        kv_shuffle.compute(global_shuffle_tensor, aclshmem_k_cache_tensor,
                        aclshmem_v_cache_tensor, src_block_tensor, dst_block_tensor)
        torch_npu.npu.synchronize() 
    else:
        print("Unknown tool, exiting...")
        return 
    
    print("pe: ", pe, " kv_shuffle end !!!!")
    npu_tensork = aclshmem_k_cache_tensor.cpu()
    npu_tensorv = aclshmem_v_cache_tensor.cpu()
    print("K are equal:", torch.equal(npu_tensork, my_cache_data.k_cache_list_g[pe]))  # True
    print("V are equal:", torch.equal(npu_tensorv, my_cache_data.v_cache_list_g[pe]))  # True

    print("K are equal may be False:", torch.equal(npu_tensork, my_cache_data.k_cache_list[pe]))  # may be False
    print("V are equal may be False:", torch.equal(npu_tensorv, my_cache_data.v_cache_list[pe]))  # may be False

    aclshmem_common.free_tensor(aclshmem_k_cache_tensor)
    aclshmem_common.free_tensor(aclshmem_v_cache_tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KVShuffle")

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

    processes = []

    for pe in range(PES):
        p = multiprocessing.Process(target=worker, args=(pe,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes have finished")