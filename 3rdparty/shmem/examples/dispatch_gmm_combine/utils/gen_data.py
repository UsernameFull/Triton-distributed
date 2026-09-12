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
import shutil
from copy import deepcopy
from enum import Enum
from collections import namedtuple
import collections
import math
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np
import numpy


LCAL_PATH = os.getcwd().replace("build", "")
DATA_PATH = os.path.join(LCAL_PATH, "utils", "test_data_golden_cpu")
shutil.rmtree(DATA_PATH, ignore_errors=True)
os.makedirs(DATA_PATH)
print(f'Use DATA_PATH = {DATA_PATH}')
print(f'Use LCAL_PATH = {LCAL_PATH}')


class CommType(Enum):
    PURE_MATMUL = 101
    ALL_REDUCE = 102
    REDUCE_SCATTER = 103
    ALL_GATHER = 104
    ALL_GATHER_V2 = 105
    MATMUL_2D = 111
    ALLTOALLV_ALLGATHER_MATMUL = 305
    MATMUL_REDUCESCATTER_ALLTOALLV = 306
    ALLTOALLVC_ALLGATHER_MATMUL = 307
    MATMUL_REDUCESCATTER_ALLTOALLVC = 308
    ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN = 309
    MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN = 310


class CoCDataTypeDesc(Enum):
    COC_DATA_TYPE_UNDEFINED = -1
    FP16FP16_FP32_FP16 = 0
    BF16BF16_FP32_BF16 = 1
    INT8INT8_INT32_FP16 = 2
    INT8INT8_INT32_BF16 = 3
    FP16INT8_INT32_FP16 = 4
    BF16INT8_INT32_BF16 = 5
    FP16INT8_FP32_FP16 = 6
    BF16INT8_FP32_BF16 = 7
    FP16INT4_FP32_FP16 = 8
    BF16INT4_FP32_BF16 = 9
    MAX = 10


CoCDataType = namedtuple('CoCDataType',
                         ['activation_dtype', 'weight_dtype', 'l0c_dtype', 'output_dtype', 'l0c_dtype_low'])

supported_coc_data_type_dict = {
    CoCDataTypeDesc.FP16FP16_FP32_FP16: CoCDataType(torch.float16, torch.float16, torch.float32, torch.float16,
                                                    torch.float16),
    CoCDataTypeDesc.BF16BF16_FP32_BF16: CoCDataType(torch.bfloat16, torch.bfloat16, torch.float32, torch.bfloat16,
                                                    torch.bfloat16),
    CoCDataTypeDesc.INT8INT8_INT32_FP16: CoCDataType(torch.int8, torch.int8, torch.int32, torch.float16, torch.float16),
    CoCDataTypeDesc.INT8INT8_INT32_BF16: CoCDataType(torch.int8, torch.int8, torch.int32, torch.bfloat16,
                                                     torch.bfloat16),
    CoCDataTypeDesc.FP16INT8_FP32_FP16: CoCDataType(torch.float16, torch.int8, torch.float32, torch.float16,
                                                    torch.float16),
    CoCDataTypeDesc.BF16INT8_FP32_BF16: CoCDataType(torch.bfloat16, torch.int8, torch.float32, torch.bfloat16,
                                                    torch.bfloat16),
}


class QuantGranularity(Enum):
    QUANT_GRANULARITY_UNDEFINED = -1
    PER_TENSOR = 0
    PER_CHANNEL = 1
    PER_GROUP = 2
    PER_TOKEN = 3
    FLOAT32_SCALE_PER_CHANNEL = 4


def generate_random_tensor(size, dtype):
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        return torch.randn(size=size, dtype=dtype)
    elif dtype is torch.int8:
        return torch.randint(-16, 16, size=size, dtype=dtype)
    elif dtype is torch.int32:
        return torch.randint(-1024, 1024, size=size, dtype=dtype)
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def generate_random_tensor_one(size, dtype):
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        return torch.ones(size=size, dtype=dtype)
    elif dtype is torch.int8:
        return torch.ones(size=size, dtype=dtype)
    elif dtype is torch.int32:
        return torch.ones(size=size, dtype=dtype)
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def convert_nd_to_nz_fpbp16(src):
    batch, k, n = src.shape
    src = src.reshape(k, n)
    c0s = 16
    k_loop = (k + c0s - 1) // c0s
    n_loop = (n + c0s - 1) // c0s
    k_align = k_loop * c0s
    n_align = n_loop * c0s
    src_pad = torch.nn.functional.pad(src, (0, n_align - n, 0, k_align - k))
    nz_w = src_pad.reshape(k_align, n_loop, c0s).permute(1, 0, 2)
    return nz_w


def convert_nd_to_nz_int8(src):
    batch, k, n = src.shape
    src = src.reshape(k, n)
    c0s = 16
    c0s2 = 32
    k_loop = (k + c0s - 1) // c0s
    n_loop = (n + c0s2 - 1) // c0s2
    k_align = k_loop * c0s
    n_align = n_loop * c0s2
    src_pad = torch.nn.functional.pad(src, (0, n_align - n, 0, k_align - k))
    nz_w = src_pad.reshape(k_align, n_loop, c0s2).permute(1, 0, 2)
    return nz_w


def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1


class QuantInfo:
    def __init__(self, rank_size, local_expert_nums, m, n, k,
                 quant_granularity=QuantGranularity.QUANT_GRANULARITY_UNDEFINED, quant_group_size=None,
                 has_quant_offset=False,
                 dequant_granularity=QuantGranularity.QUANT_GRANULARITY_UNDEFINED, dequant_group_size=None,
                 has_dequant_offset=False,
                 ):
        self.quant_granularity = quant_granularity
        self.quant_group_size = quant_group_size
        self.has_quant_offset = has_quant_offset
        self.dequant_granularity = dequant_granularity
        self.dequant_group_size = dequant_group_size
        self.has_dequant_offset = has_dequant_offset

        self.dequant_scale_origin = None
        self.dequant_args_shape = None

        self.quant_scale = None
        self.quant_offset = None
        self.dequant_scale = None
        self.dequant_offset = None

        self.rank_size = rank_size
        self.expert_per_rank = local_expert_nums
        self.m = m
        self.n = n
        self.k = k

        self.dequant_scale_list = []
        self.dequant_offset_list = []
        self.dequant_scale_origin_list = []

    def get_quant_args_shape(self, shape_info):
        m = shape_info[0]
        n = shape_info[1]
        granularity = self.dequant_granularity
        group_size = self.dequant_group_size
        if granularity is QuantGranularity.PER_TENSOR:
            return 1, 1
        elif granularity in [QuantGranularity.PER_CHANNEL, QuantGranularity.FLOAT32_SCALE_PER_CHANNEL]:
            return 1, n
        elif granularity is QuantGranularity.PER_GROUP:
            return math.ceil(m / group_size), n
        elif granularity is QuantGranularity.PER_TOKEN:
            return m, 1
        else:
            raise ValueError(f"Invalid granularity: {granularity}")

    def broadcast_quant_args(self, quant_arg, shape_info):
        granularity = self.dequant_granularity
        m = shape_info[0]
        n = shape_info[1]
        group_size = self.dequant_group_size
        if granularity is QuantGranularity.PER_GROUP:
            return quant_arg.repeat_interleave(group_size, dim=0)[:m]
        else:
            return quant_arg.expand(m, n)

    def get_pertoken_quant_tensor(self, input_info):
        shape_info = [input_info[0], input_info[2]]
        quant_args_shape = self.get_quant_args_shape(shape_info)
        self.quant_scale = generate_random_tensor(size=quant_args_shape, dtype=torch.float32) / 127
        broadcast_quant_scale = self.broadcast_quant_args(self.quant_scale, shape_info)
        return broadcast_quant_scale

    def get_output_dequant_tensor(self, input_info, l0c_dtype, coc_dtype_desc, type=0):
        # W8A8, output dequant
        shape_info = [input_info[0], input_info[2]]
        is_per_token = 0
        if self.dequant_granularity is QuantGranularity.PER_TOKEN:
            self.dequant_granularity = QuantGranularity.FLOAT32_SCALE_PER_CHANNEL
            is_per_token = 1

        # per channel
        dequant_args_shape = self.get_quant_args_shape(shape_info)
        self.dequant_args_shape = dequant_args_shape
        self.dequant_scale_origin = generate_random_tensor(size=dequant_args_shape, dtype=torch.float32) / 127
        if type:
            self.dequant_scale_origin = torch.ones(size=dequant_args_shape, dtype=torch.float32)

        if coc_dtype_desc is CoCDataTypeDesc.INT8INT8_INT32_BF16 and self.dequant_granularity in [
            QuantGranularity.FLOAT32_SCALE_PER_CHANNEL, QuantGranularity.PER_TOKEN]:
            self.dequant_scale = self.dequant_scale_origin
        else:
            self.dequant_scale_origin = ((self.dequant_scale_origin.view(torch.int32) >> 13) << 13).view(torch.float32)
            self.dequant_scale = torch.zeros(size=dequant_args_shape, dtype=torch.int64)
            self.dequant_scale.view(torch.float32)[:, ::2] = self.dequant_scale_origin

        broadcast_scale = self.broadcast_quant_args(self.dequant_scale_origin, shape_info)
        if self.has_dequant_offset == 1:
            self.dequant_offset = generate_random_tensor(size=dequant_args_shape, dtype=l0c_dtype)
            broadcast_offset = self.broadcast_quant_args(self.dequant_offset, shape_info)
        else:
            broadcast_offset = torch.zeros(dequant_args_shape, dtype=l0c_dtype)
        if is_per_token:
            self.dequant_granularity = QuantGranularity.PER_TOKEN
        return broadcast_offset, broadcast_scale

    def get_moe_dequant_tensor(self, input_info, l0c_dtype, coc_dtype_desc, type=0):
        shape_info = deepcopy(input_info)
        shape_info[-1] = shape_info[-1] * self.expert_per_rank
        self.dequant_scale_list.clear()
        self.dequant_offset_list.clear()
        self.dequant_scale_origin_list.clear()
        for _ in range(self.rank_size):
            _, _ = self.get_output_dequant_tensor(shape_info, l0c_dtype, coc_dtype_desc, type)
            self.dequant_scale_list.append(self.dequant_scale)
            self.dequant_scale_origin_list.append(self.dequant_scale_origin)
            self.dequant_scale = None
            self.dequant_scale_origin = None
            if self.has_dequant_offset == 1:
                self.dequant_offset_list.append(self.dequant_offset)

    def get_moe_broadcast_tensor(self, tp, matrix_a_block_list, l0c_dtype):
        broadcast_scale_list = []
        broadcast_offset_list = []
        for i in range(self.rank_size):
            ep_idx = i // tp
            if self.dequant_scale_list[ep_idx].shape != torch.Size([1, self.expert_per_rank * self.n]):
                dequant_scale = self.dequant_scale_origin_list[ep_idx].expand(1, self.n * self.expert_per_rank)
            else:
                dequant_scale = self.dequant_scale_origin_list[ep_idx]
            scale_blocks = torch.chunk(dequant_scale, self.expert_per_rank, dim=1)
            temp_list = []
            for j, block in enumerate(scale_blocks):
                expanded_block = block.unsqueeze(0).expand(matrix_a_block_list[i][j], -1, -1)
                temp_list.append(expanded_block.squeeze(1))
            broadcast_scale_list.append(torch.cat(temp_list, dim=0))
        if self.dequant_offset_list:
            print("!" * 30, self.dequant_offset_list)
            for i in range(self.rank_size):
                ep_idx = i // tp
                if self.dequant_offset_list[ep_idx].shape != torch.Size([1, self.expert_per_rank * self.n]):
                    dequant_offset = self.dequant_offset_list[ep_idx].expand(1, self.n * self.expert_per_rank)
                else:
                    dequant_offset = self.dequant_offset_list[ep_idx]
                offset_blocks = torch.chunk(dequant_offset, self.expert_per_rank, dim=1)
                temp_list = []
                for j, block in enumerate(offset_blocks):
                    expanded_block = block.unsqueeze(0).expand(matrix_a_block_list[i][j], -1, -1)
                    temp_list.append(expanded_block.squeeze(1))
                broadcast_offset_list.append(torch.cat(temp_list, dim=0))
        else:
            for i in range(self.rank_size):
                broadcast_offset = torch.zeros_like(broadcast_scale_list[i], dtype=l0c_dtype)
                broadcast_offset_list.append(broadcast_offset)
        return broadcast_scale_list, broadcast_offset_list

    def get_moe_pertoken_quant_tensor(self, input_info, rank_size):
        quant_tensor_list = []
        for _ in range(rank_size):
            self.get_pertoken_quant_tensor(input_info)
            self.quant_scale = self.quant_scale.unsqueeze(0)
            quant_tensor_list.append(self.quant_scale)
        return quant_tensor_list


def read_binary_file(file_path, dtype=torch.float32):
    try:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        with open(file_path, "rb") as f:
            binary_data = f.read()
        if len(binary_data) == 0:
            print(f"文件为空: {file_path}")
            return torch.tensor([], dtype=dtype)
        writable_data = bytearray(binary_data)
        tensor = torch.frombuffer(writable_data, dtype=dtype)
        return tensor
    except FileNotFoundError:
        print(f"The file {file_path} does not exist!")
        return None


class MoeTestDate:
    def __init__(self, comm_type, rank_size, batch_size, m, k, n, trans_a, trans_b, expert_per_rank,
                 coc_dtype_desc: CoCDataTypeDesc, quant_info: QuantInfo, ep, tp, weight_nz,
                 p_value, mode, max_output_size, top_k, active_num, capacity, drop_pad_mode,
                 expert_tokens_before_capacity_flag, expert_tokens_count_or_cumsum_flag,
                 quant_mode):
        self.k2 = n // 2
        self.n2 = k

        activation_dtype, weight_dtype, l0c_dtype, output_dtype, l0c_dtype_low = supported_coc_data_type_dict[
            coc_dtype_desc]
        self.matrix_a_list = []
        self.matrix_b1_list = []
        self.matrix_b2_list = []

        self.permuted_token_list = []
        self.per_token_scale2_list = []
        for _ in range(rank_size):
            self.matrix_a_list.append(generate_random_tensor(size=(m, k), dtype=torch.float16))
            self.matrix_b1_list.append(generate_random_tensor(size=(expert_per_rank, k, n), dtype=weight_dtype))
            self.matrix_b2_list.append(generate_random_tensor(size=(expert_per_rank, self.k2, self.n2),
                                       dtype=weight_dtype))
        self.expert_num = expert_per_rank * ep
        self.expert_per_rank = expert_per_rank
        self.sequence_length = m
        self.input_info = [m * top_k, k, n]
        self.batch_size = batch_size
        self.max_output_size = max_output_size
        self.trans_b = trans_b
        self.m = m
        self.k = k
        self.n = n
        self.top_k = top_k
        self.rank_size = rank_size
        self.coc_dtype_desc = coc_dtype_desc
        self.tp = tp
        self.ep = ep
        self.l0c_dtype = l0c_dtype
        self.output_dtype = output_dtype
        self.weight_nz = weight_nz
        self.p_value = p_value
        self.quant_info = quant_info

        self.endfix = f"{coc_dtype_desc.value}_{batch_size}_{m}_{k}_{n}_{expert_per_rank}_{ep}_{tp}.bin"
        if comm_type in [CommType.ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN]:
            init_routing_matrix_a = []
            num_local_tokens_per_expert = []
            self.pertoken_scale_list = []
            self.expanded_row_idx_list = []
            for i in range(rank_size):
                expert_idx = torch.randint(0, self.expert_num, (m, top_k), dtype=torch.int32)
                self.write_to_bin(expert_idx, f"expert_idx_{i}")

                print(self.matrix_a_list[i].to('npu'))
                (matrix_a, expanded_row_idx,
                 expert_tokens_count_or_cumsum, pertoken_scale) = torch_npu.npu_moe_init_routing_v2(
                    self.matrix_a_list[i].to('npu'), expert_idx.to('npu'), scale=None, offset=None,
                    active_num=m * top_k, expert_capacity=m * top_k, expert_num=self.expert_num,
                    drop_pad_mode=drop_pad_mode,
                    expert_tokens_num_type=1, expert_tokens_num_flag=True,
                    active_expert_range=[0, self.expert_num], quant_mode=quant_mode, row_idx_type=0)
                matrix_a = matrix_a.cpu().numpy()
                pertoken_scale = pertoken_scale.cpu().numpy()
                expert_tokens_count_or_cumsum = expert_tokens_count_or_cumsum.cpu().numpy()
                expanded_row_idx = expanded_row_idx.cpu().numpy()

                print(f"matrix_a shape is {matrix_a.shape}")
                self.expanded_row_idx_list.append(expanded_row_idx)
                self.write_to_bin(torch.from_numpy(matrix_a).unsqueeze(0), f"matrix_a_tmp_{i}")
                self.write_to_bin(torch.from_numpy(pertoken_scale).unsqueeze(0), f"matrix_pertoken_scale1_{i}")
                init_routing_matrix_a.append(torch.from_numpy(matrix_a).unsqueeze(0))
                num_local_tokens_per_expert.append(expert_tokens_count_or_cumsum)
                self.pertoken_scale_list.append(torch.from_numpy(pertoken_scale).unsqueeze(0).unsqueeze(2))
                print(f"self.pertoken_scale_list[{i}] shape is {self.pertoken_scale_list[i].shape}")
            (self.input_splits, self.output_splits,
             self.num_local_tokens_per_expert,
             self.num_global_tokens_per_local_expert) = self.get_moe_input_output_splits(expert_per_rank, ep, mode,
                                                        max_output_size, num_local_tokens_per_expert)

            for i in range(rank_size):
                self.write_to_bin(torch.from_numpy(self.num_local_tokens_per_expert[i]), f"tokenPerExpert_{i}")
            self.matrix_a_i_list, self.matrix_a_block_list = self.alltoall_nopermute(init_routing_matrix_a,
                                                                                     k, activation_dtype, ep)
            if self.max_output_size > 0:
                for i in range(ep):
                    self.matrix_a_i_list[i] = self.matrix_a_i_list[i][:, :max_output_size, :]

            self.dispatch_gmm_swiglu()
            self.combine(l0c_dtype_low)
            if self.trans_b:
                self.matrix_b1_list = [tensor.transpose(1, 2) for tensor in self.matrix_b1_list]
                self.matrix_b2_list = [tensor.transpose(1, 2) for tensor in self.matrix_b2_list]
            for i in range(rank_size):
                self.write_to_bin(self.matrix_a_list[i], f"matrix_a_{i}")
                self.write_to_bin(self.matrix_b1_list[i], f"matrix_b1_origin_{i}")
                self.write_to_bin(self.matrix_b2_list[i], f"matrix_b2_origin_{i}")

                if weight_nz:
                    matrix_b1 = self.convert_nd_to_nz(coc_dtype_desc, self.matrix_b1_list[i])
                    self.write_to_bin(matrix_b1, f"matrix_b1_{i}")
                    matrix_b2 = self.convert_nd_to_nz(coc_dtype_desc, self.matrix_b2_list[i])
                    self.write_to_bin(matrix_b2, f"matrix_b2_{i}")
                else:
                    self.write_to_bin(self.matrix_b1_list[i], f"matrix_b1_{i}")
                    self.write_to_bin(self.matrix_b2_list[i], f"matrix_b2_{i}")


    def get_num_local_tokens_per_expert(self, mode=1):
        if mode == 1:
            numpy.random.seed(0)
            indices = numpy.random.randint(self.expert_num, size=self.sequence_length)
            item_dict = collections.Counter(indices)
            num_local_tokens_per_expert = [item_dict.get(i, 0) for i in range(self.expert_num)]
        elif mode == 0:
            elements = [i for i in range(self.expert_num)]
            indices = elements * (self.sequence_length // self.expert_num)
            item_dict = collections.Counter(indices)
            num_local_tokens_per_expert = [item_dict.get(i, 0) for i in range(self.expert_num)]
        else:
            p = np.zeros(self.expert_num)
            p[0] = 0.9
            p[1:] = 0.1 / (self.expert_num - 1)
            indices = numpy.random.choice(self.expert_num, size=self.sequence_length, p=p)
            item_dict = collections.Counter(indices)
            num_local_tokens_per_expert = [item_dict.get(i, 0) for i in range(self.expert_num)]
        return num_local_tokens_per_expert, indices

    def write_npu_output(self, tensor, prefix):
        file_path = f"{DATA_PATH}/{prefix}.bin"
        if tensor is None:
            return
        untyped_dict = {
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16,
            torch.int8: torch.int8,
            torch.float32: torch.int32,
            torch.int32: torch.int32,
            torch.int64: torch.int64
        }
        print(tensor.shape, tensor.dtype, file_path)
        tensor.view(untyped_dict[tensor.dtype]).numpy().tofile(file_path)

    def write_to_bin(self, tensor, prefix):
        file_path = f"{DATA_PATH}/{prefix}_{self.endfix}"
        if tensor is None:
            return
        untyped_dict = {
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16,
            torch.int8: torch.int8,
            torch.float32: torch.int32,
            torch.int32: torch.int32,
            torch.int64: torch.int64
        }
        print(tensor.shape, tensor.dtype, file_path)
        tensor.view(untyped_dict[tensor.dtype]).numpy().tofile(file_path)

    def get_moe_input_output_splits(self, expert_per_rank, ep, mode, max_output_size, num_local_tokens_per_expert):
        all_gather_res = num_local_tokens_per_expert[0].tolist()
        for i in range(1, ep):
            all_gather_res = all_gather_res + num_local_tokens_per_expert[i].tolist()
        input_splits = [None] * ep
        for i in range(ep):
            input_splits[i] = numpy.sum(numpy.array(num_local_tokens_per_expert[i]).reshape(ep, expert_per_rank),
                                        axis=1)
        self.global_tokens_per_expert_matrix = numpy.array(num_local_tokens_per_expert).reshape(
                                               ep * ep * expert_per_rank)
        output_splits = [None] * ep
        num_global_tokens_per_expert = numpy.array(all_gather_res).reshape(ep, self.expert_num)
        num_global_tokens_per_local_expert = [None] * ep
        for i in range(ep):
            num_global_tokens_per_local_expert[i] = num_global_tokens_per_expert[:,
                                                    i * expert_per_rank:(i + 1) * expert_per_rank]
            output_splits[i] = numpy.sum(num_global_tokens_per_local_expert[i], axis=-1)
            self.write_to_bin(
                torch.tensor(num_local_tokens_per_expert[i]).reshape(1, ep * expert_per_rank).to(dtype=torch.int32),
                f"num_local_tokens_per_expert_{i}")

        self.write_to_bin(
            torch.from_numpy(numpy.array(num_local_tokens_per_expert)).reshape(ep, ep * expert_per_rank).to(
                dtype=torch.int32), "global_tokens_per_expert_matrix")
        return input_splits, output_splits, num_local_tokens_per_expert, num_global_tokens_per_local_expert

    def alltoall_permute(self, matrix_a, k, element_type, ep):
        m_matrix_a = [sum(self.input_splits[i]) for i in range(ep)]
        matrix_a_i_list = [torch.zeros(size=(self.batch_size, m_matrix_a[i], k), dtype=element_type) for i in range(ep)]
        matrix_a_block_list = [[] for _ in range(ep)]
        for src_ep in range(ep):
            src_offset = 0

            for local_expert_idx in range(self.expert_per_rank):
                src_offset_old = src_offset
                expert_idx = local_expert_idx + src_ep * self.expert_per_rank
                for dst_ep in range(ep):
                    dst_expert_offset = 0
                    dst_expert_len = self.num_local_tokens_per_expert[dst_ep][expert_idx]
                    for i in range(expert_idx):
                        dst_expert_offset += self.num_local_tokens_per_expert[dst_ep][i]
                    matrix_a_i_list[dst_ep][:, dst_expert_offset:dst_expert_offset + dst_expert_len, :] = (
                        matrix_a[src_ep][:, src_offset:src_offset + dst_expert_len, :])
                    src_offset += dst_expert_len

        return matrix_a_i_list

    def alltoall_nopermute(self, matrix_a, k, element_type, ep):
        m_matrix_a = [sum(self.output_splits[i]) for i in range(ep)]
        matrix_a_i_list = [torch.zeros(size=(self.batch_size, m_matrix_a[i], k), dtype=element_type) for i in range(ep)]
        matrix_a_block_list = [[] for _ in range(ep)]
        for src_ep in range(ep):
            src_offset = 0
            sum_tokens = 0
            for local_expert_idx in range(self.expert_per_rank):
                src_offset_old = src_offset
                expert_idx = local_expert_idx + src_ep * self.expert_per_rank
                for dst_ep in range(ep):
                    dst_expert_offset = 0
                    dst_expert_len = self.num_local_tokens_per_expert[dst_ep][expert_idx]
                    for i in range(expert_idx):
                        dst_expert_offset += self.num_local_tokens_per_expert[dst_ep][i]
                    matrix_a_i_list[src_ep][:, src_offset:src_offset + dst_expert_len, :] = (
                        matrix_a[dst_ep][:, dst_expert_offset:dst_expert_offset + dst_expert_len, :])
                    src_offset += dst_expert_len
                    if self.max_output_size > 0:
                        if (sum_tokens + self.global_tokens_per_expert_matrix[
                            dst_ep * self.expert_num + expert_idx]) >= self.max_output_size:
                            self.global_tokens_per_expert_matrix[
                                dst_ep * self.expert_num + expert_idx] = self.max_output_size - sum_tokens
                            sum_tokens = self.max_output_size
                        else:
                            sum_tokens += self.global_tokens_per_expert_matrix[dst_ep * self.expert_num + expert_idx]
                if self.max_output_size > 0:
                    if src_offset >= self.max_output_size and src_offset_old <= self.max_output_size:
                        src_offset = self.max_output_size
                matrix_a_block_list[src_ep].append(src_offset - src_offset_old)
        return matrix_a_i_list, matrix_a_block_list

    def convert_nd_to_nz(self, coc_dtype_desc, input_tensor):
        split_tensors = torch.unbind(input_tensor, dim=0)
        split_tensors = [t.unsqueeze(0) for t in split_tensors]
        processed_tensors = []
        for tensor in split_tensors:
            if coc_dtype_desc in [CoCDataTypeDesc.FP16FP16_FP32_FP16, CoCDataTypeDesc.BF16BF16_FP32_BF16]:
                processed_tensor = convert_nd_to_nz_fpbp16(tensor)
            else:
                processed_tensor = convert_nd_to_nz_int8(tensor)
            processed_tensors.append(processed_tensor)
        output_tensor = torch.cat(processed_tensors, dim=0)
        return output_tensor

    def swiglu(self, x: torch.Tensor) -> torch.Tensor:
        x0, gate = x.chunk(2, dim=-1)
        swish = x0 * torch.sigmoid(x0)
        y = swish * gate
        return y

    def quant(self, x: torch.Tensor):
        x_row_max = torch.max(torch.abs(x), dim=-1).values
        quant_result = x * 127. / x_row_max[:, None]
        y = torch.round(quant_result).to(torch.int8)
        scale = (x_row_max / 127.).to(torch.float32)
        return y, scale

    def unpermute(self, permuted_tokens, origin_sorted_indices, probs):
        orgin_dtype = permuted_tokens.dtype
        permuted_tokens = permuted_tokens.to(torch.float).cpu()
        sorted_indices = origin_sorted_indices.cpu()

        if probs is not None:
            probs = probs.cpu()
            num_unpermuted_tokens = probs.numel()
            topk = probs.size(1)
            probs = probs.to(torch.float)
        else:
            probs = None
            num_unpermuted_tokens = permuted_tokens.size(0)
            topk = 1

        unpermuted_tokens = torch.zeros(
            [num_unpermuted_tokens, permuted_tokens.shape[-1]],
            dtype=torch.float,
            device=permuted_tokens.device,
        )

        sorted_indices = sorted_indices.to(torch.int64)
        unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
        unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))

        if probs is not None:
            unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1)

        return unpermuted_tokens.to(orgin_dtype)

    def dispatch_gmm_swiglu(self):
        if self.coc_dtype_desc in [CoCDataTypeDesc.FP16FP16_FP32_FP16, CoCDataTypeDesc.BF16BF16_FP32_BF16]:
            for i in range(self.rank_size):
                ep_idx = i // self.tp
                a_blocks = torch.split(self.matrix_a_i_list[ep_idx], self.matrix_a_block_list[ep_idx], dim=1)
                b_blocks = torch.unbind(self.matrix_b1_list[i], dim=0)
                result_blocks = []

                for a_block, b_block in zip(a_blocks, b_blocks):
                    a_block = a_block.unsqueeze(1)
                    b_block = b_block.unsqueeze(0)
                    product = torch.matmul(a_block.to(self.l0c_dtype), b_block.to(self.l0c_dtype)).squeeze(1)
                    result_blocks.append(product)
                matrix_c = torch.cat(result_blocks, dim=1).to(self.l0c_dtype)

        elif self.coc_dtype_desc in [CoCDataTypeDesc.INT8INT8_INT32_FP16, CoCDataTypeDesc.INT8INT8_INT32_BF16]:
            if self.quant_info.dequant_granularity not in [QuantGranularity.PER_CHANNEL,
                                                      QuantGranularity.PER_TENSOR,
                                                      QuantGranularity.PER_TOKEN,
                                                      QuantGranularity.FLOAT32_SCALE_PER_CHANNEL]:
                print("error:invalid dequant_granularity: ", self.quant_info.dequant_granularity)
                return

            self.quant_info.get_moe_dequant_tensor(self.input_info, self.l0c_dtype, self.coc_dtype_desc, 0)
            dequant_scale_list = self.quant_info.dequant_scale_list
            dequant_offset_list = self.quant_info.dequant_offset_list
            broadcast_scale_list, broadcast_offset_list = self.quant_info.get_moe_broadcast_tensor(self.tp,
                                                          self.matrix_a_block_list, self.l0c_dtype)
            for i in range(self.rank_size):
                if dequant_offset_list:
                    self.write_to_bin(dequant_offset_list[i], f"matrix_dequant_offset1_{i}")
                self.write_to_bin(dequant_scale_list[i], f"matrix_dequant_scale1_{i}")
                self.write_to_bin(self.quant_info.dequant_scale_origin_list[i], f"matrix_dequant_scale1_origin_{i}")

            if self.quant_info.dequant_granularity is QuantGranularity.PER_TOKEN:
                quant_scale_list = self.pertoken_scale_list
                print("@" * 20, quant_scale_list[0].shape)
                quant_scale_alltoall, _ = self.alltoall_nopermute(quant_scale_list, 1, torch.float32, self.ep)
                for i in range(self.rank_size):
                    ep_idx = i // self.tp
                    quant_scale = quant_scale_list[ep_idx].squeeze(0)
                    quant_scale_alltoall[ep_idx] = quant_scale_alltoall[ep_idx].squeeze(0)
                    if self.max_output_size > 0:
                        quant_scale_alltoall[ep_idx] = quant_scale_alltoall[ep_idx][:self.max_output_size, :]

            for i in range(self.rank_size):
                ep_idx = i // self.tp
                a_blocks = torch.split(self.matrix_a_i_list[ep_idx], self.matrix_a_block_list[ep_idx], dim=1)
                b_blocks = torch.unbind(self.matrix_b1_list[i], dim=0)
                result_blocks = []
                for a_block, b_block in zip(a_blocks, b_blocks):
                    a_block = a_block.unsqueeze(1)
                    b_block = b_block.unsqueeze(0)
                    product = torch.matmul(a_block.to(self.l0c_dtype), b_block.to(self.l0c_dtype)).squeeze(1)
                    result_blocks.append(product)
                matrix_c = torch.cat(result_blocks, dim=1).to(self.l0c_dtype)

                matrix_c = ((matrix_c + broadcast_offset_list[i]).to(torch.float32) * broadcast_scale_list[i]).to(
                    torch.float16)
                self.write_to_bin(matrix_c.to(torch.float16), f"matrix_c_{i}")

                if self.quant_info.dequant_granularity is QuantGranularity.PER_TOKEN:
                    broadcast_quant_scale = quant_scale_alltoall[ep_idx].expand(-1, self.input_info[2])
                    matrix_c = (matrix_c.to(torch.float32) * broadcast_quant_scale)

                swiglu_out = self.swiglu(matrix_c.squeeze(0))
                premuted_token, per_token_scale2 = self.quant(swiglu_out)
                self.permuted_token_list.append(premuted_token.to(torch.float16).to(torch.int8))
                self.per_token_scale2_list.append(per_token_scale2.to(torch.float32))
                self.write_to_bin(premuted_token, f"matrix_permuted_token_{i}")
                self.write_to_bin(per_token_scale2, f"matrix_pertoken_scale2_{i}")

    def combine(self, l0c_dtype_low):
        input_info = [self.m * self.top_k, self.k2, self.n2]

        origin_sorted_indecies = []
        for i in range(self.rank_size):
            origin_sorted_indecies.append(torch.from_numpy(self.expanded_row_idx_list[i]).to(torch.int32))

        probs = torch.randn(size=(self.m, self.top_k), dtype=torch.float32)
        self.write_to_bin(probs, f"probs")

        all_matrix_b2_list_per_expert = []
        for i in range(self.rank_size):
            ep_idx = i // self.tp
            b_blocks = torch.unbind(self.matrix_b2_list[ep_idx], dim=0)
            all_matrix_b2_list_per_expert += b_blocks

        if self.coc_dtype_desc in [CoCDataTypeDesc.INT8INT8_INT32_FP16, CoCDataTypeDesc.INT8INT8_INT32_BF16]:
            if self.quant_info.dequant_granularity not in [QuantGranularity.PER_CHANNEL,
                                                      QuantGranularity.PER_TENSOR,
                                                      QuantGranularity.PER_TOKEN,
                                                      QuantGranularity.FLOAT32_SCALE_PER_CHANNEL]:
                print("error:invalid dequant_granularity: ", self.quant_info.dequant_granularity)
                return

            self.quant_info.get_moe_dequant_tensor(input_info, self.l0c_dtype, self.coc_dtype_desc, 0)
            dequant_scale_list = self.quant_info.dequant_scale_list
            dequant_offset_list = self.quant_info.dequant_offset_list
            dequant_scale_origin_list = self.quant_info.dequant_scale_origin_list

            scale_list_per_tensor = []
            offset_list_per_tensor = []
            for i in range(self.rank_size):
                if dequant_offset_list:
                    self.write_to_bin(dequant_offset_list[i], f"matrix_dequant_offset2_{i}")
                    split_offset = torch.split(dequant_offset_list[i], input_info[2], dim=1)
                else:
                    dequant_offset = torch.zeros_like(dequant_scale_list[i], dtype=self.l0c_dtype)
                    split_offset = torch.split(dequant_offset, input_info[2], dim=1)
                print(f"dequant_scale_list[i] shape is {dequant_scale_list[i].shape} "
                      f"type is {dequant_scale_list[i].dtype}")
                self.write_to_bin(dequant_scale_list[i], f"matrix_dequant_scale2_{i}")
                self.write_to_bin(self.quant_info.dequant_scale_origin_list[i], f"matrix_dequant_scale2_origin_{i}")
                split_scale = torch.split(dequant_scale_origin_list[i], input_info[2], dim=1)

                offset_list_per_tensor.extend(split_offset)
                scale_list_per_tensor.extend(split_scale)
            matrix_c_list = []

            activation_dtype, weight_dtype, self.l0c_dtype, self.output_dtype, l0c_dtype_low = (
                supported_coc_data_type_dict[self.coc_dtype_desc])

            permuted_token_list = []

            for i in range(self.rank_size):
                permuted_token_list.append(self.permuted_token_list[i].unsqueeze(0))
                print(f"permuted_token_list.shape shape is {permuted_token_list[i].shape}")

            matrix_a_i_list = self.alltoall_permute(permuted_token_list, self.k2, activation_dtype, self.ep)

            for i in range(self.rank_size):
                ep_idx = i // self.tp
                global_actual_token = self.global_tokens_per_expert_matrix[
                    i * self.ep * self.expert_per_rank:(i + 1) * self.ep * self.expert_per_rank].tolist()
                print(f"matrix_a_i_list[{ep_idx}] shape is {matrix_a_i_list[ep_idx].shape}")
                a_blocks = torch.split(matrix_a_i_list[ep_idx],
                                       self.num_local_tokens_per_expert[ep_idx].tolist(), dim=1)
                result_blocks = []
                for j, _ in enumerate(a_blocks):
                    a_block = a_blocks[j].unsqueeze(1)
                    b_block = all_matrix_b2_list_per_expert[j].unsqueeze(0)
                    broadcast_offset = offset_list_per_tensor[j]
                    broadcast_scale = scale_list_per_tensor[j]
                    product = torch.matmul(a_block.to(torch.float32), b_block.to(torch.float32)).squeeze(1).to(
                        self.l0c_dtype)
                    matrix_c_out = ((product + broadcast_offset).to(torch.float32) * (broadcast_scale)).to(
                        torch.float32)
                    result_blocks.append(matrix_c_out)
                matrix_c = torch.cat(result_blocks, dim=1)
                tmp_offset = 0
                for t, _ in enumerate(global_actual_token):
                    if self.num_local_tokens_per_expert[ep_idx][t] != global_actual_token[t]:
                        left = tmp_offset + global_actual_token[t]
                        right = tmp_offset + self.num_local_tokens_per_expert[ep_idx][t]
                        matrix_c[:, left:right, :] = 0
                    tmp_offset += self.num_local_tokens_per_expert[ep_idx][t]
                matrix_c_list.append(matrix_c)

            if self.quant_info.dequant_granularity is QuantGranularity.PER_TOKEN:
                per_token_scale2_list = []
                for i in range(self.rank_size):
                    per_token_scale2_list.append(self.per_token_scale2_list[i].unsqueeze(0).unsqueeze(2))

                quant_scale_list = self.alltoall_permute(per_token_scale2_list, 1, torch.float32, self.ep)
                for i in range(self.rank_size):
                    ep_idx = i // self.tp
                    broadcast_quant_scale = quant_scale_list[ep_idx]
                    matrix_c_list[ep_idx] = (matrix_c_list[ep_idx] * broadcast_quant_scale).to(torch.float32)

            for i in range(self.rank_size):
                ep_idx = i // self.tp
                permuted_tokens = matrix_c_list[i].to(self.output_dtype)
                self.write_to_bin(permuted_tokens, f"ptrC2_{ep_idx}")
                self.write_npu_output(torch_npu.npu_moe_token_unpermute(permuted_tokens.squeeze(0).to('npu'),
                                  origin_sorted_indecies[ep_idx].to('npu'),
                                  probs.to('npu')).cpu().to(torch.float32), f"unpermuted_token_{ep_idx}")


def validate_args(data_type):
    coc_dtype_desc = CoCDataTypeDesc(data_type)
    if coc_dtype_desc not in supported_coc_data_type_dict:
        raise ValueError(f'Unsupported CoC data type {coc_dtype_desc}')
    print(f'Use CoC data type: {str(coc_dtype_desc)}')


def main():
    import configparser
    config = configparser.ConfigParser()
    config.read(os.path.join(LCAL_PATH, './utils/config.ini'))
    comm_type = int(config['global']['cocType'])
    data_type = int(config['global']['dataType'])
    rank_size = int(config['global']['rankSize'])
    batch = int(config['mmInfo']['batchSize'])
    m = int(config['mmInfo']['m'])
    k = int(config['mmInfo']['k'])
    n = int(config['mmInfo']['n'])
    trans_a = int(config['mmInfo']['transA'])
    trans_b = int(config['mmInfo']['transB'])
    bias = int(config['mmInfo']['withBias'])
    weight_nz = int(config['mmInfo']['weightNz'])
    rmsnorm = int(config['PostInfo']['withRmsNorm'])
    quant_granularity = int(config['quantInfo']['quantGranularity'])
    quant_group_size = int(config['quantInfo']['quantGroupSize'])
    has_quant_offset = int(config['quantInfo']['hasQuantOffset'])
    dequant_granularity = int(config['quantInfo']['dequantGranularity'])
    dequant_group_size = int(config['quantInfo']['dequantGroupSize'])
    has_dequant_offset = int(config['quantInfo']['hasDequantOffset'])
    p_value = int(config['tiling']['pValue'])

    local_expert_nums = int(config['moeInfo']['local_expert_nums'])
    ep = int(config['moeInfo']['EP'])
    tp = int(config['moeInfo']['TP'])
    mode = int(config['moeInfo']['mode'])
    max_output_size = int(config['moeInfo']['maxOutputSize'])
    top_k = int(config['initRoutingInfo']['topK'])
    active_num = int(config['initRoutingInfo']['activeNum'])
    capacity = int(config['initRoutingInfo']['expertCapacity'])
    drop_pad_mode = int(config['initRoutingInfo']['dropPadMode'])
    expert_tokens_before_capacity_flag = config['initRoutingInfo']['expertTokensBeforeCapacityFlag']
    expert_tokens_count_or_cumsum_flag = int(config['initRoutingInfo']['expertTokensCountOrCumsumFlag'])
    quant_mode = int(config['initRoutingInfo']['quantMode'])

    is_deterministic = os.environ.get('LCCL_DETERMINISTIC')
    if is_deterministic is not None and is_deterministic.lower() in ['true', '1']:
        is_deterministic = 1
    else:
        is_deterministic = 0

    validate_args(data_type)

    quant_info = QuantInfo(rank_size, local_expert_nums, m, n, k, QuantGranularity(quant_granularity),
                           quant_group_size, has_quant_offset, QuantGranularity(dequant_granularity),
                           dequant_group_size, has_dequant_offset)


    MoeTestDate(CommType(comm_type), rank_size, batch, m, k, n, trans_a, trans_b, local_expert_nums,
                CoCDataTypeDesc(data_type), quant_info, ep, tp, weight_nz, p_value, mode, max_output_size,
                top_k, active_num, capacity, drop_pad_mode, expert_tokens_before_capacity_flag,
                expert_tokens_count_or_cumsum_flag, quant_mode)

if __name__ == '__main__':
    main()
