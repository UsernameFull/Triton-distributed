from abc import ABC, abstractmethod
from typing import Any, Optional
import torch
import torch.distributed
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu

COMM_STREAM = None

def async_all_to_all(input_,
                     output_split_sizes,
                     input_split_sizes,
                     group,
                     event=None):
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(
                device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True)
    else:
        handle = dist.all_to_all_single(a2a_out,
                                        input_.contiguous(),
                                        output_split_sizes=output_split_sizes,
                                        input_split_sizes=input_split_sizes,
                                        group=group,
                                        async_op=True)
    return input_, a2a_out, handle

def _gather_along_first_dim(input_, group, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 NPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size,
                             dtype=input_.dtype,
                             device=torch.npu.current_device())
        torch.distributed.all_gather_into_tensor(output,
                                                 input_.contiguous(),
                                                 group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size,
                             dtype=input_.dtype,
                             device=torch.npu.current_device())
        output_tensor_list = list(
            torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def gather_from_sequence_parallel_region(
    input_,
    group,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _gather_along_first_dim(input_, group, output_split_sizes)


class MoETokenDispatcher(ABC):

    def __init__(self, **kwargs) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.top_k = kwargs.get("top_k", 0)
        self.num_experts = kwargs.get("num_experts", 0)

    @abstractmethod
    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[Any] = None,
                       quantized_x_for_share: Optional[Any] = None,
                       dynamic_scale_for_share: Optional[Any] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
        raise NotImplementedError("Combine function not implemented.")

class TokenDispatcherWithAll2AllV(MoETokenDispatcher):
    """
    The implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.with_quant = False
        self.num_local_experts = kwargs.get("num_local_experts", 0)
        self.ep_rank = kwargs.get("rank", 0)
        self.ep_size = kwargs.get("ep_size", 0)
        self.ep_group = kwargs.get("ep_group", 0)
        self.num_experts = kwargs.get("num_experts", 0)

        self.hidden_shape = None
        self.topk_weights = None
        self.input_splits = None
        self.output_splits = None
        self.hidden_shape_before_permute = None

        # [tp_ep_size * ep_size, num_local_experts]. Represents the number of tokens sent
        # to each local expert by all ranks.
        self.num_global_tokens_per_local_expert = None

        # cached intermediate tensors.
        self.tokens_per_expert = None
        self.global_input_tokens_local_experts_indices = None

        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.npu.current_device(),
            )

        local_expert_indices_offset = (self.ep_rank * self.num_local_experts)

        self.local_expert_indices = [
            local_expert_indices_offset + i
            for i in range(self.num_local_experts)
        ]
        assert (len(self.local_expert_indices) == self.num_local_experts
                ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (self.local_expert_indices[i] ==
                    self.local_expert_indices[i + 1] -
                    1), "local_expert_indices must be continuous"

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[Any] = None,
                       quantized_x_for_share: Optional[Any] = None,
                       dynamic_scale_for_share: Optional[Any] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
        self.with_quant = with_quant
        self.hidden_shape = hidden_states.shape
        self.topk_weights = topk_weights
        assert topk_weights.dim() == 2, "Expected 2D tensor for topk_weights"
        assert topk_ids.dim() == 2, "Expected 2D tensor for routing map"

        permutated_local_input_tokens, reversed_local_input_permutation_mapping, tokens_per_expert = self._dispatch_preprocess(
            hidden_states, topk_ids)
        self.reversed_local_input_permutation_mapping = reversed_local_input_permutation_mapping

        dynamic_scale_after_all2all = None

        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            self.ep_group,
        )
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        global_input_tokens, dynamic_scale = self._dispatch_postprocess(
            global_input_tokens, dynamic_scale_after_all2all)

        return {
            "hidden_states": global_input_tokens,
            "group_list": tokens_per_expert,
            "dynamic_scale": dynamic_scale,
            "group_list_type": 1
        }

    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        hidden_states = self._combine_preprocess(hidden_states)

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states, self.input_splits, self.output_splits,
            self.ep_group)
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        output = self._combine_postprocess(permutated_local_input_tokens)

        # these values are no longer used, so they need to be set to None for memory release.
        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None
        self.topk_weights = None
        self.reversed_local_input_permutation_mapping = None
        self.reversed_global_input_permutation_mapping = None
        self.global_input_tokens_local_experts_indices = None

        return output

    def _dispatch_preprocess(self, hidden_states, topk_ids):
        assert self.hidden_shape is not None
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self._preprocess(topk_ids)

        self.hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=self.num_out_tokens,
        )
        return permutated_local_input_tokens, reversed_local_input_permutation_mapping, tokens_per_expert

    def _preprocess(self, topk_ids: torch.Tensor) -> torch.Tensor:
        num_local_tokens_per_expert = torch.histc(topk_ids,
                                                  bins=self.num_experts,
                                                  min=0,
                                                  max=self.num_experts)

        ep_size = self.ep_size

        self.num_out_tokens = topk_ids.numel()

        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        self.input_splits = (num_local_tokens_per_expert.reshape(
            ep_size,
            self.num_local_experts).sum(axis=1).to(torch.device("cpu"),
                                                   non_blocking=True).numpy())
        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert,
            group=self.ep_group).reshape(ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices[
            0]:self.local_expert_indices[-1] + 1]
        if self.num_global_tokens_per_local_expert is None:
            raise ValueError(
                "num_global_tokens_per_local_expert must be set before sum.")
        self.output_splits = (self.num_global_tokens_per_local_expert.sum(
            axis=-1).to(torch.device("cpu"), non_blocking=True).numpy())
        num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(
            axis=0)

        if self.num_local_experts > 1:
            if self.num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                self.num_global_tokens_per_local_expert.ravel())
        else:
            torch.npu.synchronize()

        return num_tokens_per_local_expert

    def _dispatch_postprocess(self, global_input_tokens, dynamic_scale=None):
        # Early return if no local experts or no tokens
        if self.num_local_experts <= 1:
            return global_input_tokens, None

        # Handle quantized case
        if self.with_quant:
            assert self.global_input_tokens_local_experts_indices is not None, \
            "global_input_tokens_local_experts_indices must be initialized before calling _dispatch_postprocess"
            expert_idx_2d = self.global_input_tokens_local_experts_indices.unsqueeze(
                -1)
            active_num = self.global_input_tokens_local_experts_indices.numel()

            # Handle case with no active tokens
            if active_num <= 0:
                self.reversed_global_input_permutation_mapping = self.global_input_tokens_local_experts_indices
                return global_input_tokens, dynamic_scale

            # Process with active tokens
            global_input_tokens, self.reversed_global_input_permutation_mapping, _, expanded_scale = torch_npu.npu_moe_init_routing_v2(
                global_input_tokens,
                expert_idx_2d,
                scale=None,
                active_num=active_num,
                expert_capacity=active_num,
                expert_num=self.num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.num_experts],
                quant_mode=1,
                row_idx_type=0)
            return global_input_tokens, expanded_scale

        # Handle non-quantized case
        global_input_tokens, self.reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens,
            self.global_input_tokens_local_experts_indices)
        return global_input_tokens, None

    def _combine_preprocess(self, hidden_states):
        # Unpermutation 2: expert output to AlltoAll input
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1:
            hidden_states = torch_npu.npu_moe_token_unpermute(
                hidden_states, self.reversed_global_input_permutation_mapping)

        return hidden_states

    def _combine_postprocess(self, permutated_local_input_tokens):
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=self.reversed_local_input_permutation_mapping.to(
                torch.int32),
            probs=self.topk_weights,
            restore_shape=self.hidden_shape_before_permute)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output