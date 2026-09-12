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
import torch
import numpy as np

# Model Params
GROUP_SZIE = 4
MAX_SEQLEN = 1024
MAX_BATCH = 10
INIT_BATCH = 5

# KVCache Params
PAGE_SIZE = 128
max_block_nums = MAX_SEQLEN * MAX_BATCH // PAGE_SIZE
KV_HEAD_NUM = 8
HEAD_DIM = 128

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def get_pair_pe(sort_idx, local_pe):
    pair_pe = 0
    for idx, pe in enumerate(sort_idx):
        if pe == local_pe:
            pair_idx = (GROUP_SZIE - 1) - idx # notice idx can't be GROUP_SZIE
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
    for i in range(GROUP_SZIE):
        pair_idx = get_pair_pe(sort_idx, i)
        pair_list.append([pair_idx])

    # Get pe to pe transfer_tokens
    transfer_tokens_list = []
    for i in range(GROUP_SZIE):
        transfer_tokens, transfer_batch_id = get_pair_transfer_tokens(kv_lens, kv_sum, kv_mean, pair_list, i)
        if (transfer_tokens > 0):
            pair_list[i].append(0) # 0 means send
        else:
            pair_list[i].append(1) # 1 means recv
        transfer_tokens_list.append((transfer_tokens, transfer_batch_id))
    return pair_list, transfer_tokens_list


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pe_size', type=int)
    args = parser.parse_args()

    global GROUP_SZIE
    GROUP_SZIE = args.pe_size

    torch.manual_seed(42)
    kv_lens = torch.randint(0, MAX_SEQLEN, (GROUP_SZIE, INIT_BATCH))
    kv_sum = torch.sum(kv_lens, dim=-1) # (pe_size, 1)

    k_cache_list = []
    v_cache_list = []
    used_blocks_list = []
    batch_blocks_list = []
    # Prepare Inputs
    for i in range(GROUP_SZIE):
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
    golden_path = os.path.join(SCRIPT_PATH, "output")
    if not os.path.exists(golden_path):
        os.mkdir(golden_path)
    for i in range(GROUP_SZIE):
        k_path = os.path.join(SCRIPT_PATH, "output", f"k_cache_input_pe_{i}.bin")
        v_path = os.path.join(SCRIPT_PATH, "output", f"v_cache_input_pe_{i}.bin")
        k_cache_list[i].numpy().tofile(k_path)
        v_cache_list[i].numpy().tofile(v_path)

    src_block_table = []
    dst_block_table = []
    block_num_list = []
    # Params Prepare And Golden Calculate
    for i in range(GROUP_SZIE):
        local_pe = i
        shuffle_table = pair_list
        k_cache_list = k_cache_list
        v_cache_list = v_cache_list
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
        if (shuffle_table[local_pe][1] == 0):
            for idx, _ in enumerate(src_local_table):
                src_idx = src_local_table[idx]
                dst_idx = dst_local_table[idx]
                k_cache_list[pair_pe][dst_idx] = k_cache_list[local_pe][src_idx]
                v_cache_list[pair_pe][dst_idx] = v_cache_list[local_pe][src_idx]

    # Record pair_list
    pair_list_path = os.path.join(SCRIPT_PATH, "output", f"pair_list.bin")
    np.array(pair_list).tofile(pair_list_path)

    # Record src_block_table
    for i in range(GROUP_SZIE):
        src_block_table_path = os.path.join(SCRIPT_PATH, "output", f"src_block_table_pe_{i}.bin")
        if src_block_table[i] is not None:
            np.array(src_block_table[i]).tofile(src_block_table_path)

    # Record dst_block_table
    for i in range(GROUP_SZIE):
        dst_block_table_path = os.path.join(SCRIPT_PATH, "output", f"dst_block_table_pe_{i}.bin")
        if dst_block_table[i] is not None:
            np.array(dst_block_table[i]).tofile(dst_block_table_path)

    # Record block_nums
    for i in range(GROUP_SZIE):
        block_nums_path = os.path.join(SCRIPT_PATH, "output", f"block_num_pe_{i}.bin")
        np.array(block_num_list[i]).tofile(block_nums_path)

    # Record Output KV Cache
    for i in range(GROUP_SZIE):
        k_path = os.path.join(SCRIPT_PATH, "output", f"k_cache_golden_pe_{i}.bin")
        v_path = os.path.join(SCRIPT_PATH, "output", f"v_cache_golden_pe_{i}.bin")
        k_cache_list[i].numpy().tofile(k_path)
        v_cache_list[i].numpy().tofile(v_path)

if __name__ == "__main__":
    main()