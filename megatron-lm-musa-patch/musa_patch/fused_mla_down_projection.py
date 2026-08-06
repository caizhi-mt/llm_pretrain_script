"""Fuse MLA q/kv down projections without changing checkpoint parameters.

The two projection weights remain separate parameters so existing checkpoints,
optimizer state, and parameter names stay unchanged. Forward and dgrad use a
temporary packed weight and one GEMM; wgrad still writes the two original FP32
``main_grad`` buffers with the existing MUSA accumulation kernel.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn.functional as F


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def is_supported(hidden_states, q_module, kv_module, config) -> bool:
    if not _env_flag("MUSA_FUSED_MLA_DOWN_PROJ", "1"):
        return False
    if getattr(config, "q_lora_rank", None) is None:
        return False
    if getattr(config, "tensor_model_parallel_size", 1) != 1:
        return False
    if hidden_states.dtype not in {torch.float16, torch.bfloat16}:
        return False
    if hidden_states.device.type != "musa":
        return False

    q_weight = getattr(q_module, "weight", None)
    kv_weight = getattr(kv_module, "weight", None)
    if q_weight is None or kv_weight is None:
        return False
    if getattr(config, "add_bias_linear", True):
        return False
    if q_weight.dtype != hidden_states.dtype or kv_weight.dtype != hidden_states.dtype:
        return False
    if (
        q_weight.device != hidden_states.device
        or kv_weight.device != hidden_states.device
    ):
        return False
    if q_weight.dim() != 2 or kv_weight.dim() != 2:
        return False
    if (
        q_weight.shape[1] != hidden_states.shape[-1]
        or kv_weight.shape[1] != hidden_states.shape[-1]
    ):
        return False
    return q_weight.is_contiguous() and kv_weight.is_contiguous()


def _prepare_wgrad_tensors(inp: torch.Tensor, grad_output: torch.Tensor):
    from megatron.core.tensor_parallel.layers import (
        prepare_input_tensors_for_wgrad_compute,
    )

    return prepare_input_tensors_for_wgrad_compute(grad_output, inp)


def _accumulate_main_grad(
    inp: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    main_grad = getattr(weight, "main_grad", None)
    grad_output, inp = _prepare_wgrad_tensors(inp, grad_output)
    if main_grad is None:
        return torch.mm(grad_output.T, inp)

    import fused_weight_gradient_mlp_cuda

    if main_grad.dtype == torch.float32:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
            inp, grad_output, main_grad
        )
    elif main_grad.dtype in (torch.float16, torch.bfloat16):
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
            inp, grad_output, main_grad
        )
    else:
        raise RuntimeError(f"unsupported main_grad dtype: {main_grad.dtype}")

    if hasattr(weight, "grad_added_to_main_grad"):
        weight.grad_added_to_main_grad = True
    if getattr(weight, "zero_out_wgrad", False):
        return torch.empty_like(weight)
    return None


class _FusedMLADownProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, q_weight, kv_weight):
        q_rank = q_weight.shape[0]
        packed_weight = torch.cat((q_weight, kv_weight), dim=0)
        with torch.profiler.record_function("musa_fused_mla_down_proj_fwd"):
            packed_output = F.linear(hidden_states, packed_weight)

        ctx.save_for_backward(hidden_states, q_weight, kv_weight, packed_weight)
        return packed_output[..., :q_rank], packed_output[..., q_rank:]

    @staticmethod
    def backward(ctx, q_grad, kv_grad):
        hidden_states, q_weight, kv_weight, packed_weight = ctx.saved_tensors
        if not q_grad.is_contiguous():
            q_grad = q_grad.contiguous()
        if not kv_grad.is_contiguous():
            kv_grad = kv_grad.contiguous()

        with torch.profiler.record_function("musa_fused_mla_down_proj_dgrad"):
            packed_grad = torch.cat((q_grad, kv_grad), dim=-1)
            hidden_grad = torch.matmul(packed_grad, packed_weight)

        with torch.profiler.record_function("musa_fused_mla_down_proj_wgrad"):
            q_weight_grad = _accumulate_main_grad(hidden_states, q_grad, q_weight)
            kv_weight_grad = _accumulate_main_grad(hidden_states, kv_grad, kv_weight)

        return hidden_grad, q_weight_grad, kv_weight_grad


def fused_mla_down_projection(
    hidden_states: torch.Tensor,
    q_module,
    kv_module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _FusedMLADownProjection.apply(
        hidden_states,
        q_module.weight,
        kv_module.weight,
    )
