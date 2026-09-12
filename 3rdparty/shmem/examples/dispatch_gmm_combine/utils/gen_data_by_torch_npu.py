import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import shutil
from enum import Enum
from collections import namedtuple
from copy import deepcopy
from token_dispatch_combine import TokenDispatcherWithAll2AllV

import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*leaked.*")

LCAL_PATH = os.getcwd().replace("build", "")
CPU_DATA_PATH = LCAL_PATH + "/utils/test_data_golden_cpu/"
DATA_PATH = os.path.join(LCAL_PATH, "utils", "test_data_torch_npu")
os.makedirs(DATA_PATH, exist_ok=True)

def read_binary_file(file_path, dtype=torch.float32):
    try:
        with open(file_path, "rb") as f:
            binary_data = f.read()
        writable_data = bytearray(binary_data)
        tensor = torch.frombuffer(writable_data, dtype=dtype)
        return tensor
    except FileNotFoundError:
        print(f"The file {file_path} does not exist!")
        return None

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

def convert_nd_to_nz(input_tensor):
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

class QuantInfo:
    def __init__(self, quant_granularity=QuantGranularity.QUANT_GRANULARITY_UNDEFINED, quant_group_size=None,
                 has_quant_offset=False,
                 dequant_granularity=QuantGranularity.QUANT_GRANULARITY_UNDEFINED, dequant_group_size=None,
                 has_dequant_offset=False):
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

    def get_output_dequant_tensor(self, input_info, l0c_dtype, coc_dtype_desc, TYPE=0):
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
        if TYPE :
            self.dequant_scale_origin = torch.ones(size=dequant_args_shape, dtype=torch.float32)

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

    def get_moe_dequant_tensor(self, rank_size, expert_per_rank, input_info, l0c_dtype, coc_dtype_desc, TYPE=0):
        shape_info = deepcopy(input_info)
        shape_info[-1] = shape_info[-1] * expert_per_rank
        self.dequant_scale_list = []
        self.dequant_offset_list = []
        self.dequant_scale_origin_list = []
        for _ in range(rank_size):
            _, _ = self.get_output_dequant_tensor(shape_info, l0c_dtype, coc_dtype_desc, TYPE)
            self.dequant_scale_list.append(self.dequant_scale)
            self.dequant_scale_origin_list.append(self.dequant_scale_origin)
            self.dequant_scale = None
            self.dequant_scale_origin = None
            if self.has_dequant_offset == 1:
                self.dequant_offset_list.append(self.dequant_offset)


def write_to_bin(tensor, prefix, endfix):
    file_path = f"{DATA_PATH}/{prefix}_{endfix}"
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

def get_new_group(rank, ep_world_size):
    tp_world_size = 1
    for i in range(tp_world_size):
        # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
        ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group_t = ep_group
            print(f"rank:{rank} ep_ranks:{ep_ranks}")
    for i in range(ep_world_size):
        # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
        tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
        tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
        if rank in tp_ranks:
            tp_group_t = tp_group
            print(f"rank:{rank} tp_ranks:{tp_ranks}")
    return ep_group_t, tp_group_t

def get_hcomm_info(rank, comm_group):
    if torch.__version__ > '2.0.1':
        hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = comm_group.get_hccl_comm_name(rank)
    return hcomm_info

def quant_apply_mlp(rank,
                    hidden_states: torch.Tensor,
                    w1: torch.Tensor,
                    w1_scale: torch.Tensor,
                    w2: torch.Tensor,
                    w2_scale: torch.Tensor,
                    group_list: torch.Tensor,
                    group_list_type: int = 1,
                    dynamic_scale: torch.Tensor = None,
                    output_dtype=torch.float16) -> torch.Tensor:
    if dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
            hidden_states)
    else:
        pertoken_scale = dynamic_scale

    _output_dtype = output_dtype

    if w1_scale.dtype != torch.float32:
        w1_scale = w1_scale.to(torch.float32)
    if w2_scale.dtype != torch.float32:
        w2_scale = w2_scale.to(torch.float32)

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=3,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.int32)[0]
    write_npu_output(hidden_states.cpu(), f"matrix_c_{rank}")
    hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
        x=hidden_states,
        weight_scale=w1_scale,
        activation_scale=pertoken_scale,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=group_list,
        activate_left=True,
        quant_mode=1,
    )
    print("!" * 50, hidden_states, hidden_states.shape)
    write_npu_output(hidden_states.cpu(), f"swiglu_{rank}")
    write_npu_output(swiglu_out_scale.cpu(), f"matrix_pertoken_scale2_{rank}")

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale],
        per_token_scale=[swiglu_out_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=_output_dtype)[0]

    write_npu_output(hidden_states.cpu(), f"cpu_before_commbine_{rank}")
    return hidden_states


def unquant_apply_mlp(hidden_states: torch.Tensor,
                      w1: torch.Tensor,
                      w2: torch.Tensor,
                      group_list: torch.Tensor,
                      group_list_type: int = 1,
                      topk_scales: torch.Tensor = None,
                      need_trans: bool = True) -> torch.Tensor:

    if need_trans:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    if topk_scales is not None:
        gate_up_out *= topk_scales

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    return hidden_states

def write_npu_output(tensor, prefix):
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


def fused_experts(
        rank,
        ep_group,
        ep_hcomm_info,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        m,
        k,
        n,
        top_k, expert_per_rank, world_size, dequant_granularity, drop_pad_mode, quant_mode,
        # For load balance
        log2phy: torch.Tensor = None,
        need_trans: bool = False):
    # Check constraints
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    _output_dtype = hidden_states.dtype
    with_quant = False
    if quant_mode:
        with_quant = True
    token_dispatcher = TokenDispatcherWithAll2AllV(num_local_experts=expert_per_rank, num_experts=expert_per_rank*world_size, rank=rank, ep_size=world_size, ep_group=ep_group)

    results = token_dispatcher.token_dispatch(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        with_quant=with_quant)

    permuted_hidden_states, expert_tokens, dynamic_scale, group_list_type, topk_scales = \
        results["hidden_states"], results["group_list"], results.get("dynamic_scale"), results[
            "group_list_type"], results.get("topk_scales")

    write_npu_output(permuted_hidden_states.cpu(), f"dispatch_{rank}")
    quant_type = dequant_granularity
    if quant_type:
        mlp_output = quant_apply_mlp(rank, hidden_states=permuted_hidden_states,
                               w1=w1,
                               w1_scale=w1_scale,
                               w2=w2,
                               w2_scale=w2_scale,
                               group_list=expert_tokens,
                               group_list_type=group_list_type,
                               dynamic_scale=dynamic_scale,
                               output_dtype=_output_dtype)
    else:
        mlp_output = unquant_apply_mlp(hidden_states=permuted_hidden_states,
                                 w1=w1,
                                 w2=w2,
                                 group_list=expert_tokens,
                                 group_list_type=group_list_type,
                                 topk_scales=topk_scales,
                                 need_trans=need_trans)

    final_hidden_states = token_dispatcher.token_combine(hidden_states=mlp_output)

    return final_hidden_states


def get_input_tensor_from_bin():
    matrix_a_list = []
    matrix_b1_list = []
    matrix_b2_list = []
    expert_idx_list = []
    dequant_scale1_list = []
    dequant_scale2_list = []
    for i in range(rank_size):
        matrix_a = read_binary_file(f"{CPU_DATA_PATH}/matrix_a_{i}_{coc_dtype_desc.value}_{batch_size}_{m}_{k}_{n}_{expert_per_rank}_{EP}_{TP}.bin", output_dtype)
        matrix_b1 = read_binary_file(f"{CPU_DATA_PATH}/matrix_b1_origin_{i}_{coc_dtype_desc.value}_{batch_size}_{m}_{k}_{n}_{expert_per_rank}_{EP}_{TP}.bin", weight_dtype)
        matrix_b2 = read_binary_file(f"{CPU_DATA_PATH}/matrix_b2_origin_{i}_{coc_dtype_desc.value}_{batch_size}_{m}_{k}_{n}_{expert_per_rank}_{EP}_{TP}.bin", weight_dtype)
        expert_idx = read_binary_file(f"{CPU_DATA_PATH}/expert_idx_{i}_{coc_dtype_desc.value}_{batch_size}_{m}_{k}_{n}_{expert_per_rank}_{EP}_{TP}.bin", torch.int32)
        matrix_dequant_scale1 = read_binary_file(
            f"{CPU_DATA_PATH}/matrix_dequant_scale1_origin_{i}_{coc_dtype_desc.value}_{batch_size}_{m}_{k}_{n}_{expert_per_rank}_{EP}_{TP}.bin",
            torch.float32)
        matrix_dequant_scale2 = read_binary_file(
            f"{CPU_DATA_PATH}/matrix_dequant_scale2_origin_{i}_{coc_dtype_desc.value}_{batch_size}_{m}_{k}_{n}_{expert_per_rank}_{EP}_{TP}.bin",
            torch.float32)
        matrix_a_list.append(matrix_a.reshape(m,k))
        matrix_b1_list.append(matrix_b1.reshape(expert_per_rank, k, n))
        matrix_b2_list.append(matrix_b2.reshape(expert_per_rank, k2, n2))
        expert_idx_list.append(expert_idx.reshape(m, top_k))
        dequant_scale1_list.append(matrix_dequant_scale1)
        dequant_scale2_list.append(matrix_dequant_scale2)
    probs = read_binary_file(f"{CPU_DATA_PATH}/probs_{coc_dtype_desc.value}_{batch_size}_{m}_{k}_{n}_{expert_per_rank}_{EP}_{TP}.bin", torch.float32).reshape(m, top_k)

    return matrix_a_list, matrix_b1_list, matrix_b2_list, expert_idx_list, probs, dequant_scale1_list, dequant_scale2_list


def generate_input_tensor():
    matrix_a_list = []
    matrix_b1_list = []
    matrix_b2_list = []
    expert_idx_list = []
    for i in range(rank_size):
        matrix_a_list.append(generate_random_tensor(size=(m, k), dtype=output_dtype))
        matrix_b1_list.append(generate_random_tensor(size=(expert_per_rank, k, n), dtype=weight_dtype))
        matrix_b2_list.append(generate_random_tensor(size=(expert_per_rank, k2, n2), dtype=weight_dtype))
        expert_idx_list.append(torch.randint(0, expert_num, (m, top_k), dtype=torch.int32))
    probs = torch.randn(size=(m, top_k), dtype=torch.float32)
    quant_info = QuantInfo(QuantGranularity(quant_granularity), quant_group_size, has_quant_offset,
                           QuantGranularity(dequant_granularity), dequant_group_size, has_dequant_offset)
    assert quant_info.dequant_granularity in [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_TENSOR,
                                              QuantGranularity.PER_TOKEN,
                                              QuantGranularity.FLOAT32_SCALE_PER_CHANNEL]
    quant_info.get_moe_dequant_tensor(rank_size, expert_per_rank, input_info, l0c_dtype, coc_dtype_desc, 0)
    dequant_scale1_list = quant_info.dequant_scale_list
    dequant_scale1_origin_list = quant_info.dequant_scale_origin_list
    dequant_offset1_list = quant_info.dequant_offset_list
    shape_info = [m * top_k, k2, n2]
    quant_info.get_moe_dequant_tensor(rank_size, expert_per_rank, shape_info, l0c_dtype,
                                      coc_dtype_desc, 0)
    dequant_scale2_list = quant_info.dequant_scale_list
    dequant_scale2_origin_list = quant_info.dequant_scale_origin_list
    dequant_offset2_list = quant_info.dequant_offset_list

    return matrix_a_list, matrix_b1_list, matrix_b2_list, expert_idx_list, probs, dequant_scale1_list, dequant_scale2_list, dequant_scale1_origin_list, dequant_scale2_origin_list

def gen_golen(rank, rank_size, m, k, n, expert_per_rank, top_k, dequant_granularity, drop_pad_mode, quant_mode, matrix_a_list, matrix_b1_list, matrix_b2_list, expert_idx_list, probs, dequant_scale1_list, dequant_scale2_list):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + '127.0.0.1' + ':' + '60001'
    try:
        dist.init_process_group(backend="hccl", rank=rank, world_size=rank_size, init_method=init_method)
        _, _ = ep_group, tp_group = get_new_group(rank, rank_size)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        hidden_states = matrix_a_list[rank].npu()
        w1 = matrix_b1_list[rank].npu()
        w1_scale = dequant_scale1_list[rank].reshape(expert_per_rank, n).npu()
        w2 = matrix_b2_list[rank].npu()
        w2_scale = dequant_scale2_list[rank].reshape(expert_per_rank, k).npu()
        topk_weights = probs.npu()
        topk_ids = expert_idx_list[rank].npu()
        final_result = fused_experts(rank, ep_group, ep_hcomm_info, hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale, m, k, n, top_k, expert_per_rank, rank_size, dequant_granularity, drop_pad_mode, quant_mode, None, False)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        print(f"[INFO][Rank {rank}] Cleanup complete.")

    write_npu_output(final_result.cpu(), f"unpermuted_token_{rank}")


if __name__ == '__main__':
    import configparser

    print(f"LCAL_PATH={LCAL_PATH}, DATA_PATH={DATA_PATH}")
    config = configparser.ConfigParser()
    config.read(os.path.join(LCAL_PATH, './utils/config.ini'))
    comm_type = int(config['global']['cocType'])
    data_type = int(config['global']['dataType'])
    rank_size = int(config['global']['rankSize'])
    batch_size = int(config['mmInfo']['batchSize'])
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
    pValue = int(config['tiling']['pValue'])

    expert_per_rank = int(config['moeInfo']['local_expert_nums'])
    EP = int(config['moeInfo']['EP'])
    TP = int(config['moeInfo']['TP'])
    mode = int(config['moeInfo']['mode'])
    maxOutputSize = int(config['moeInfo']['maxOutputSize'])
    top_k = int(config['initRoutingInfo']['topK'])
    active_num = int(config['initRoutingInfo']['activeNum'])
    capacity = int(config['initRoutingInfo']['expertCapacity'])
    drop_pad_mode = int(config['initRoutingInfo']['dropPadMode'])
    expert_tokens_before_capacity_flag = config['initRoutingInfo']['expertTokensBeforeCapacityFlag']
    expert_tokens_count_or_cumsum_flag = int(config['initRoutingInfo']['expertTokensCountOrCumsumFlag'])
    quant_mode = int(config['initRoutingInfo']['quantMode'])

    coc_dtype_desc = CoCDataTypeDesc(data_type)
    activation_dtype, weight_dtype, l0c_dtype, output_dtype, l0c_dtype_low = supported_coc_data_type_dict[coc_dtype_desc]
    k2 = n // 2
    n2 = k
    expert_num = expert_per_rank * EP
    input_info = [m * top_k, k, n]


    print(mode)  # 'random'
    print(mode == 'random')

    endfix = f"{coc_dtype_desc.value}_{batch_size}_{m}_{k}_{n}_{expert_per_rank}_{EP}_{TP}.bin"

    matrix_a_list, matrix_b1_list, matrix_b2_list, expert_idx_list, probs, dequant_scale1_origin_list, dequant_scale2_origin_list = get_input_tensor_from_bin()
    print("matrix_a:", matrix_a_list[0])
    mp.spawn(gen_golen, args=(rank_size, m, k, n, expert_per_rank, top_k, dequant_granularity, drop_pad_mode, quant_mode, matrix_a_list, matrix_b1_list, matrix_b2_list, expert_idx_list, probs, dequant_scale1_origin_list, dequant_scale2_origin_list), nprocs=rank_size)

    if trans_b:
        matrix_b1_list = [tensor.transpose(1, 2) for tensor in matrix_b1_list]
        matrix_b2_list = [tensor.transpose(1, 2) for tensor in matrix_b2_list]
    for i in range(rank_size):
        write_to_bin(matrix_a_list[i], f"matrix_a_{i}", endfix)
        write_to_bin(expert_idx_list[i], f"expert_idx_{i}", endfix)

        write_to_bin(dequant_scale1_origin_list[i], f"matrix_dequant_scale1_origin_{i}", endfix)
        write_to_bin(dequant_scale2_origin_list[i], f"matrix_dequant_scale2_origin_{i}", endfix)
        if weight_nz:
            matrix_b1 = convert_nd_to_nz(matrix_b1_list[i])
            write_to_bin(matrix_b1, f"matrix_b1_{i}", endfix)
            matrix_b2 = convert_nd_to_nz(matrix_b2_list[i])
            write_to_bin(matrix_b2, f"matrix_b2_{i}", endfix)
        else:
            write_to_bin(matrix_b1_list[i], f"matrix_b1_{i}", endfix)
            write_to_bin(matrix_b2_list[i], f"matrix_b2_{i}", endfix)



