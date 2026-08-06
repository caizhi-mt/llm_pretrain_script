"""Compact DeepEP-local MoE permutation on top of TE's native MUSA kernels.

DeepEP returns a compact ``[tokens, router_topk]`` expert-index/probability
representation.  The stock TE mask-map path expands that metadata to 32 local
expert columns and its native data-movement kernels scan all 32 columns for
every token.  This module preserves TE's expert-major row assignment, compacts
the resulting row map back to ``router_topk`` columns, and then calls the same
TE MUSA kernels with the compact map.

No routing result, tensor, or data pointer is cached across calls.  Counts may
be arbitrarily non-uniform and invalid compact slots remain encoded as ``-1``.
DeepEP/router top-k indices are expected to be unique within each token.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAVE_TRITON = False


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
        "",
    }


def compact_permutation_enabled() -> bool:
    return _HAVE_TRITON and _env_flag("MUSA_COMPACT_PERMUTE", "1")


if _HAVE_TRITON:

    @triton.jit
    def _indices_to_routing_map_kernel(
        indices_ptr,
        routing_map_ptr,
        num_experts: tl.constexpr,
        num_experts_pow2: tl.constexpr,
        topk: tl.constexpr,
        topk_pow2: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        topk_offsets = tl.arange(0, topk_pow2)
        topk_mask = topk_offsets < topk
        expert_indices = tl.load(
            indices_ptr + token_idx * topk + topk_offsets,
            mask=topk_mask,
            other=-1,
        )

        expert_offsets = tl.arange(0, num_experts_pow2)
        expert_mask = expert_offsets < num_experts
        output_offsets = token_idx * num_experts + expert_offsets
        tl.store(routing_map_ptr + output_offsets, 0, mask=expert_mask)
        tl.debug_barrier()
        valid = (
            topk_mask
            & (expert_indices >= 0)
            & (expert_indices < num_experts)
        )
        tl.store(
            routing_map_ptr + token_idx * num_experts + expert_indices,
            1,
            mask=valid,
        )


    @triton.jit
    def _compact_row_maps_kernel(
        dense_row_map_ptr,
        indices_ptr,
        compact_row_map_ptr,
        compact_row_map_trans_ptr,
        num_tokens,
        num_experts: tl.constexpr,
        topk: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        total = num_tokens * topk
        offset_mask = offsets < total
        token_idx = offsets // topk
        topk_idx = offsets % topk
        expert_idx = tl.load(indices_ptr + offsets, mask=offset_mask, other=-1)
        valid = (
            offset_mask & (expert_idx >= 0) & (expert_idx < num_experts)
        )
        safe_expert_idx = tl.where(valid, expert_idx, 0)
        row_id = tl.load(
            dense_row_map_ptr + token_idx * num_experts + safe_expert_idx,
            mask=offset_mask,
            other=-1,
        )
        row_id = tl.where(valid, row_id, -1)
        tl.store(compact_row_map_ptr + offsets, row_id, mask=offset_mask)
        tl.store(
            compact_row_map_trans_ptr + topk_idx * num_tokens + token_idx,
            row_id,
            mask=offset_mask,
        )


def indices_to_routing_map(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    if not _HAVE_TRITON:
        raise RuntimeError("compact permutation requires Triton")
    num_tokens, topk = indices.shape
    routing_map = torch.empty(
        (num_tokens, num_experts), dtype=torch.bool, device=indices.device
    )
    _indices_to_routing_map_kernel[(num_tokens,)](
        indices,
        routing_map,
        num_experts=num_experts,
        num_experts_pow2=triton.next_power_of_2(num_experts),
        topk=topk,
        topk_pow2=triton.next_power_of_2(topk),
        num_warps=1,
    )
    return routing_map


def _make_compact_row_maps(
    indices: torch.Tensor,
    routing_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from transformer_engine.pytorch.triton.permutation import make_row_id_map

    num_tokens, num_experts = routing_map.shape
    topk = indices.shape[1]
    _, dense_row_map = make_row_id_map(routing_map, num_tokens, num_experts)
    compact_row_map = torch.empty(
        (num_tokens, topk), dtype=dense_row_map.dtype, device=indices.device
    )
    compact_row_map_trans = torch.empty(
        (topk, num_tokens), dtype=dense_row_map.dtype, device=indices.device
    )
    block_size = 256
    grid = (triton.cdiv(num_tokens * topk, block_size),)
    _compact_row_maps_kernel[grid](
        dense_row_map,
        indices,
        compact_row_map,
        compact_row_map_trans,
        num_tokens,
        num_experts=num_experts,
        topk=topk,
        block_size=block_size,
        num_warps=4,
    )
    return compact_row_map_trans, compact_row_map


def _empty_tensor() -> torch.Tensor:
    # TE's MUSA bindings use a zero-sized CPU tensor as the optional argument
    # sentinel, matching transformer_engine.pytorch.permutation.
    return _EMPTY_TENSOR


_EMPTY_TENSOR = torch.empty(0, device="cpu")


def _preallocated_view(
    tensor: Optional[torch.Tensor],
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    required = math.prod(shape)
    if tensor is None:
        output = torch.empty(shape, dtype=dtype, device=device)
    else:
        if tensor.device != device:
            raise ValueError(
                f"preallocated tensor must be on {device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError("preallocated tensor must be contiguous")
        storage = tensor.view(dtype).reshape(-1)
        if storage.numel() < required:
            raise ValueError(
                f"preallocated tensor has {storage.numel()} {dtype} elements, "
                f"but {required} are required"
            )
        output = storage[:required].view(shape)
    return output


def _optional_preallocated_view(
    tensor: Optional[torch.Tensor],
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    empty: torch.Tensor,
) -> torch.Tensor:
    if tensor is None:
        return empty
    return _preallocated_view(tensor, shape, dtype, device)


class _CompactPermuteWithProbs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        compact_probs: torch.Tensor,
        compact_indices: torch.Tensor,
        routing_map: torch.Tensor,
        num_out_tokens: int,
        preallocated_act_f: Optional[torch.Tensor],
        preallocated_probs_f: Optional[torch.Tensor],
        preallocated_act_b: Optional[torch.Tensor],
        preallocated_probs_b: Optional[torch.Tensor],
    ):
        from transformer_engine.pytorch import cpp_extensions as tex
        from transformer_engine.pytorch.constants import TE_DType

        num_tokens, hidden_size = inp.shape
        topk = compact_indices.shape[1]
        compact_row_map_trans, compact_row_map = _make_compact_row_maps(
            compact_indices, routing_map
        )
        empty = _empty_tensor()
        act_f_buffer = _optional_preallocated_view(
            preallocated_act_f,
            (num_out_tokens, hidden_size),
            inp.dtype,
            inp.device,
            empty,
        )
        probs_f_buffer = _optional_preallocated_view(
            preallocated_probs_f,
            (num_out_tokens,),
            compact_probs.dtype,
            compact_probs.device,
            empty,
        )
        output, permuted_probs = tex.moe_permute_mask(
            TE_DType[inp.dtype],
            inp,
            compact_row_map,
            compact_probs,
            num_tokens,
            topk,
            num_out_tokens,
            hidden_size,
            act_f_buffer,
            probs_f_buffer,
        )

        ctx.save_for_backward(compact_row_map_trans)
        ctx.num_tokens = num_tokens
        ctx.topk = topk
        ctx.hidden_size = hidden_size
        ctx.preallocated_act_b = preallocated_act_b
        ctx.preallocated_probs_b = preallocated_probs_b
        ctx.probs_dtype = compact_probs.dtype
        ctx.mark_non_differentiable(compact_row_map_trans)
        return output, permuted_probs, compact_row_map_trans

    @staticmethod
    def backward(
        ctx,
        output_grad,
        permuted_probs_grad,
        _compact_row_map_trans_grad,
    ):
        from transformer_engine.pytorch import cpp_extensions as tex
        from transformer_engine.pytorch.constants import TE_DType

        (compact_row_map_trans,) = ctx.saved_tensors
        if not output_grad.is_contiguous():
            output_grad = output_grad.contiguous()
        if permuted_probs_grad is not None and not permuted_probs_grad.is_contiguous():
            permuted_probs_grad = permuted_probs_grad.contiguous()
        empty = _empty_tensor()
        act_grad_buffer = _optional_preallocated_view(
            ctx.preallocated_act_b,
            (ctx.num_tokens, ctx.hidden_size),
            output_grad.dtype,
            output_grad.device,
            empty,
        )
        probs_grad_buffer = empty
        if permuted_probs_grad is not None:
            probs_grad_buffer = _preallocated_view(
                ctx.preallocated_probs_b,
                (ctx.num_tokens, ctx.topk),
                ctx.probs_dtype,
                output_grad.device,
            )
        act_grad, compact_probs_grad = tex.moe_unpermute_mask(
            TE_DType[output_grad.dtype],
            output_grad,
            compact_row_map_trans,
            empty,
            permuted_probs_grad if permuted_probs_grad is not None else empty,
            ctx.num_tokens,
            ctx.topk,
            ctx.hidden_size,
            act_grad_buffer,
            probs_grad_buffer,
        )
        if permuted_probs_grad is None:
            compact_probs_grad = None
        return (
            act_grad,
            compact_probs_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _CompactUnpermute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        compact_row_map_trans: torch.Tensor,
        restore_tokens: int,
        preallocated_act_f: Optional[torch.Tensor],
        preallocated_act_b: Optional[torch.Tensor],
    ):
        from transformer_engine.pytorch import cpp_extensions as tex
        from transformer_engine.pytorch.constants import TE_DType

        empty = _empty_tensor()
        hidden_size = inp.shape[1]
        topk = compact_row_map_trans.shape[0]
        act_f_buffer = _optional_preallocated_view(
            preallocated_act_f,
            (restore_tokens, hidden_size),
            inp.dtype,
            inp.device,
            empty,
        )
        output, _ = tex.moe_unpermute_mask(
            TE_DType[inp.dtype],
            inp,
            compact_row_map_trans,
            empty,
            empty,
            restore_tokens,
            topk,
            hidden_size,
            act_f_buffer,
            empty,
        )
        # Keep the expert-major map for backward as well.  TE selects its
        # lower-overhead ``permute_with_mask_map_trans`` kernel when the first
        # dimension equals ``num_experts``.  Saving the token-major map here
        # would produce the same values, but would force the slower generic
        # map-copy kernel in backward.
        ctx.save_for_backward(compact_row_map_trans)
        ctx.restore_tokens = restore_tokens
        ctx.topk = topk
        ctx.num_out_tokens = inp.shape[0]
        ctx.hidden_size = hidden_size
        ctx.preallocated_act_b = preallocated_act_b
        return output

    @staticmethod
    def backward(ctx, output_grad):
        from transformer_engine.pytorch import cpp_extensions as tex
        from transformer_engine.pytorch.constants import TE_DType

        (compact_row_map_trans,) = ctx.saved_tensors
        if not output_grad.is_contiguous():
            output_grad = output_grad.contiguous()
        empty = _empty_tensor()
        act_grad_buffer = _optional_preallocated_view(
            ctx.preallocated_act_b,
            (ctx.num_out_tokens, ctx.hidden_size),
            output_grad.dtype,
            output_grad.device,
            empty,
        )
        act_grad, _ = tex.moe_permute_mask(
            TE_DType[output_grad.dtype],
            output_grad,
            compact_row_map_trans,
            empty,
            ctx.restore_tokens,
            ctx.topk,
            ctx.num_out_tokens,
            ctx.hidden_size,
            act_grad_buffer,
            empty,
        )
        return act_grad, None, None, None, None


def compact_permute_with_probs(
    inp: torch.Tensor,
    compact_probs: torch.Tensor,
    compact_indices: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
    *,
    preallocated_act_f: Optional[torch.Tensor] = None,
    preallocated_probs_f: Optional[torch.Tensor] = None,
    preallocated_act_b: Optional[torch.Tensor] = None,
    preallocated_probs_b: Optional[torch.Tensor] = None,
):
    output, permuted_probs, compact_row_map_trans = (
        _CompactPermuteWithProbs.apply(
            inp,
            compact_probs,
            compact_indices,
            routing_map,
            num_out_tokens,
            preallocated_act_f,
            preallocated_probs_f,
            preallocated_act_b,
            preallocated_probs_b,
        )
    )
    return output, permuted_probs, compact_row_map_trans


def compact_unpermute(
    inp: torch.Tensor,
    compact_row_map_trans: torch.Tensor,
    restore_shape: torch.Size,
    *,
    preallocated_act_f: Optional[torch.Tensor] = None,
    preallocated_act_b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    output = _CompactUnpermute.apply(
        inp,
        compact_row_map_trans,
        restore_shape[0],
        preallocated_act_f,
        preallocated_act_b,
    )
    return output.view(restore_shape)


def is_supported(
    hidden_states: torch.Tensor,
    compact_indices: torch.Tensor,
    compact_probs: torch.Tensor,
    num_experts: int,
) -> bool:
    return (
        compact_permutation_enabled()
        and hidden_states.device.type == "musa"
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.dim() == 2
        and hidden_states.numel() > 0
        # TE's BF16 unpermute kernel reads/writes 16-byte vectors without a
        # hidden-dimension tail mask.
        and hidden_states.shape[1] % 8 == 0
        and hidden_states.is_contiguous()
        and compact_indices.device == hidden_states.device
        and compact_indices.dim() == 2
        and compact_indices.shape[0] == hidden_states.shape[0]
        and compact_indices.shape[1] > 0
        # TE distinguishes [tokens, topk] from [topk, tokens] by shape[0].
        # Avoid the ambiguous tiny-input case; production N is much larger.
        and compact_indices.shape[0] != compact_indices.shape[1]
        and compact_indices.dtype in (torch.int32, torch.int64)
        and compact_indices.is_contiguous()
        and compact_probs.device == hidden_states.device
        and compact_probs.dtype == torch.float32
        and compact_probs.shape == compact_indices.shape
        and compact_probs.is_contiguous()
        and compact_indices.shape[1] % 4 == 0
        and num_experts >= compact_indices.shape[1]
    )
