# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from ..compact_permutation import (
    compact_permute_with_probs,
    compact_unpermute,
    indices_to_routing_map,
    is_supported as compact_permutation_supported,
)

from megatron.core.transformer.moe.fused_a2a import (
    get_buffer,
    get_hidden_bytes,
)
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import (
        fused_permute_with_probs,
        fused_unpermute,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def _DeepepManager_get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor):
        deepep_buffer =  get_buffer(self.group, get_hidden_bytes(hidden_states))
        ace_hidden_states, _ = deepep_buffer.get_ace_combine_buffer(self.hidden_shape_before_permute[0], self.hidden_shape_before_permute[1], 1, False)

        if not HAVE_TE or fused_unpermute is None:
            raise ValueError("fused_unpermute is not available. Please install TE >= 2.1.0.")
        if getattr(self, "_musa_compact_permutation_active", False):
            hidden_states = compact_unpermute(
                hidden_states,
                self.reversed_mapping_for_combine,
                restore_shape=self.hidden_shape_before_permute,
                preallocated_act_f=ace_hidden_states,
            )
        else:
            hidden_states =  fused_unpermute(
                hidden_states,
                self.reversed_mapping_for_combine,
                restore_shape=self.hidden_shape_before_permute,
                preallocated_act_f=ace_hidden_states,
            )

        return hidden_states


def _DeepepManager_get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor):
    deepep_buffer =  get_buffer(self.group, get_hidden_bytes(hidden_states))

    ace_hidden_states, ace_probs = deepep_buffer.get_ace_combine_buffer(
        hidden_states.size(0), hidden_states.size(1), self.router_topk, True)

    host_splits = getattr(self.tokens_per_expert, "_mate_m_splits", None)
    compact_indices = self.dispatched_indices
    compact_probs = self.dispatched_probs
    use_compact_permutation = compact_permutation_supported(
        hidden_states,
        compact_indices,
        compact_probs,
        self.num_local_experts,
    )
    if use_compact_permutation:
        self.dispatched_routing_map = indices_to_routing_map(
            compact_indices, self.num_local_experts
        )
        self.dispatched_probs = compact_probs
    else:
        self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(
            compact_indices, compact_probs, self.num_local_experts, preallocated_probs_b=ace_probs
        )

    if getattr(self.tokens_per_expert, "_mate_deferred_device_counts", False):
        # The routing map already contains the exact per-expert membership on
        # device.  A column reduction is much cheaper than contended atomics
        # and avoids both the pageable H2D copy and a CPU-visible item() sync.
        # MUSA's large-column bool reduction is inaccurate for the current
        # dispatched-token shape (~32K x 32).  Keep each reduction below the
        # backend threshold, then reduce the tiny partial-count matrix.
        count_chunks = self.dispatched_routing_map.split(8192, dim=0)
        partial_counts = [chunk.sum(dim=0, dtype=torch.int32) for chunk in count_chunks]
        device_counts = torch.stack(partial_counts, dim=0).sum(dim=0, dtype=torch.int32)
        device_counts._mate_m_splits = host_splits
        self.tokens_per_expert = device_counts

    # if self.config.moe_router_padding_for_fp8:
    #     self.dispatched_routing_map, self.tokens_per_expert = self._pad_routing_map(
    #         self.dispatched_routing_map, self.tokens_per_expert
    #     )

    self.hidden_shape_before_permute = hidden_states.shape
    assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"

    num_out_tokens = (
        sum(host_splits) if host_splits is not None else self.tokens_per_expert.sum().item()
    )
    if use_compact_permutation:
        hidden_states, permuted_probs, self.reversed_mapping_for_combine = (
            compact_permute_with_probs(
                hidden_states,
                compact_probs,
                compact_indices,
                self.dispatched_routing_map,
                num_out_tokens=num_out_tokens,
                preallocated_act_b=ace_hidden_states,
                preallocated_probs_b=ace_probs,
            )
        )
    else:
        hidden_states, permuted_probs, self.reversed_mapping_for_combine = (
            fused_permute_with_probs(
                hidden_states,
                self.dispatched_probs,
                self.dispatched_routing_map,
                num_out_tokens=num_out_tokens,
                preallocated_act_b=ace_hidden_states,
            )
        )
    self._musa_compact_permutation_active = use_compact_permutation

    if self.router_dtype == "fp64":
        permuted_probs = permuted_probs.to(torch.float64)
    return hidden_states, permuted_probs


from transformer_engine.musa.pytorch.utils import replace_attr
from megatron.core.transformer.moe.token_dispatcher import _DeepepManager

replace_attr(_DeepepManager, 'get_restored_hidden_states_by_experts', _DeepepManager_get_restored_hidden_states_by_experts)
replace_attr(_DeepepManager, 'get_permuted_hidden_states_by_experts', _DeepepManager_get_permuted_hidden_states_by_experts)