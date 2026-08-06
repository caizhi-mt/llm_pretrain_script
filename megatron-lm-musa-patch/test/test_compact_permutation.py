"""Focused tests for the DeepEP compact permutation path."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "musa_patch" / "compact_permutation.py"
SPEC = importlib.util.spec_from_file_location("compact_permutation_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_compact_permutation_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("MUSA_COMPACT_PERMUTE", raising=False)
    assert MODULE.compact_permutation_enabled() is MODULE._HAVE_TRITON

    monkeypatch.setenv("MUSA_COMPACT_PERMUTE", "0")
    assert not MODULE.compact_permutation_enabled()


def test_cpu_tensors_fall_back_even_when_enabled(monkeypatch):
    monkeypatch.setenv("MUSA_COMPACT_PERMUTE", "1")
    hidden = torch.empty((4, 16), dtype=torch.bfloat16)
    indices = torch.zeros((4, 8), dtype=torch.int64)
    probs = torch.zeros((4, 8), dtype=torch.float32)

    assert not MODULE.is_supported(hidden, indices, probs, num_experts=32)


def test_native_kernel_shape_guards_are_explicit(monkeypatch):
    monkeypatch.setenv("MUSA_COMPACT_PERMUTE", "1")

    class FakeMusaTensor:
        def __init__(self, shape, dtype, *, contiguous=True):
            self.shape = shape
            self.dtype = dtype
            self.device = torch.device("musa")
            self.dim = lambda: len(shape)
            self.numel = lambda: math.prod(shape)
            self.is_contiguous = lambda: contiguous

    hidden = FakeMusaTensor((16, 2047), torch.bfloat16)
    indices = FakeMusaTensor((16, 8), torch.int64)
    probs = FakeMusaTensor((16, 8), torch.float32)

    assert not MODULE.is_supported(hidden, indices, probs, num_experts=32)

    hidden.shape = (16, 2048)
    assert MODULE.is_supported(hidden, indices, probs, num_experts=32) is MODULE._HAVE_TRITON

    hidden.shape = (8, 2048)
    indices.shape = (8, 8)
    probs.shape = (8, 8)
    assert not MODULE.is_supported(hidden, indices, probs, num_experts=32)


def test_optional_empty_tensor_sentinel_is_reused():
    first = MODULE._empty_tensor()
    second = MODULE._empty_tensor()

    assert first is second
    assert first.device.type == "cpu"
    assert first.numel() == 0
    assert first.data_ptr() == 0


def test_preallocated_view_reuses_storage():
    storage = torch.empty(64, dtype=torch.uint8)
    result = MODULE._preallocated_view(
        storage, (2, 8), torch.float32, torch.device("cpu")
    )

    assert result.shape == (2, 8)
    assert result.dtype == torch.float32
    assert result.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()


def test_preallocated_view_rejects_invalid_storage():
    with pytest.raises(ValueError, match="must be contiguous"):
        MODULE._preallocated_view(
            torch.empty((8, 8), dtype=torch.float32).T,
            (8, 8),
            torch.float32,
            torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="but 16 are required"):
        MODULE._preallocated_view(
            torch.empty(8, dtype=torch.float32),
            (2, 8),
            torch.float32,
            torch.device("cpu"),
        )


@pytest.mark.skipif(
    os.getenv("RUN_MUSA_TESTS", "0") != "1"
    or not hasattr(torch, "musa")
    or not torch.musa.is_available(),
    reason="set RUN_MUSA_TESTS=1 on a MUSA host",
)
def test_musa_forward_and_backward_match_dense_te_path():
    from transformer_engine.pytorch.permutation import (
        moe_permute_with_probs,
        moe_unpermute,
    )

    torch.manual_seed(1234)
    # Exercise a production-like token/hidden shape that is supported by the
    # installed TE MUSA async-copy kernel.
    tokens, hidden_size, experts, topk, valid_topk = 4096, 2048, 32, 8, 4
    logits = torch.randn(tokens, experts, device="musa")
    logits += torch.linspace(0.35, -0.35, experts, device="musa")
    values, selected = torch.topk(logits, valid_topk, dim=-1)

    indices = torch.full((tokens, topk), -1, dtype=torch.int64, device="musa")
    indices[:, :valid_topk] = selected
    base_probs = torch.zeros((tokens, topk), dtype=torch.float32, device="musa")
    base_probs[:, :valid_topk] = torch.softmax(values, dim=-1)
    routing_map = torch.zeros((tokens, experts), dtype=torch.bool, device="musa")
    routing_map.scatter_(1, selected, True)
    num_out_tokens = tokens * valid_topk

    hidden_ref = torch.randn(
        tokens, hidden_size, dtype=torch.bfloat16, device="musa", requires_grad=True
    )
    hidden_compact = hidden_ref.detach().clone().requires_grad_(True)
    probs_ref = base_probs.detach().clone().requires_grad_(True)
    probs_compact = base_probs.detach().clone().requires_grad_(True)
    dense_probs = torch.zeros(
        tokens, experts, dtype=torch.float32, device="musa"
    ).scatter(1, selected, probs_ref[:, :valid_topk])

    def raw_storage(shape, dtype):
        element_size = torch.empty((), dtype=dtype).element_size()
        return torch.empty(
            math.prod(shape) * element_size, dtype=torch.uint8, device="musa"
        )

    permuted_ref, permuted_probs_ref, row_map_ref = moe_permute_with_probs(
        hidden_ref,
        dense_probs,
        routing_map,
        num_out_tokens=num_out_tokens,
    )
    restored_ref = moe_unpermute(
        permuted_ref, row_map_ref, restore_shape=hidden_ref.shape
    )

    compact_routing_map = MODULE.indices_to_routing_map(indices, experts)
    permuted_shape = (num_out_tokens, hidden_size)
    restored_shape = (tokens, hidden_size)
    permuted_compact, permuted_probs_compact, row_map_compact = (
        MODULE.compact_permute_with_probs(
            hidden_compact,
            probs_compact,
            indices,
            compact_routing_map,
            num_out_tokens,
            preallocated_act_f=raw_storage(permuted_shape, torch.bfloat16),
            preallocated_probs_f=raw_storage((num_out_tokens,), torch.float32),
            preallocated_act_b=raw_storage(restored_shape, torch.bfloat16),
            preallocated_probs_b=raw_storage((tokens, topk), torch.float32),
        )
    )
    restored_compact = MODULE.compact_unpermute(
        permuted_compact,
        row_map_compact,
        restore_shape=hidden_compact.shape,
        preallocated_act_f=raw_storage(restored_shape, torch.bfloat16),
        preallocated_act_b=raw_storage(permuted_shape, torch.bfloat16),
    )

    outputs_ref = (permuted_ref, permuted_probs_ref, restored_ref)
    outputs_compact = (
        permuted_compact,
        permuted_probs_compact,
        restored_compact,
    )
    for reference, actual in zip(outputs_ref, outputs_compact):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    sum_loss = sum(output.sum() for output in outputs_compact)
    sum_hidden_grad, sum_probs_grad = torch.autograd.grad(
        sum_loss,
        (hidden_compact, probs_compact),
        retain_graph=True,
    )
    sum_hidden_grad = sum_hidden_grad.clone()
    sum_probs_grad = sum_probs_grad.clone()
    explicit_hidden_grad, explicit_probs_grad = torch.autograd.grad(
        outputs_compact,
        (hidden_compact, probs_compact),
        grad_outputs=tuple(torch.ones_like(output) for output in outputs_compact),
        retain_graph=True,
    )
    assert torch.isfinite(sum_hidden_grad).all()
    assert torch.isfinite(sum_probs_grad).all()
    torch.testing.assert_close(sum_hidden_grad, explicit_hidden_grad, rtol=0, atol=0)
    torch.testing.assert_close(sum_probs_grad, explicit_probs_grad, rtol=0, atol=0)

    grad_outputs = tuple(torch.randn_like(output) for output in outputs_ref)
    hidden_grad_ref, probs_grad_ref = torch.autograd.grad(
        outputs_ref, (hidden_ref, probs_ref), grad_outputs=grad_outputs
    )
    hidden_grad_compact, probs_grad_compact = torch.autograd.grad(
        outputs_compact,
        (hidden_compact, probs_compact),
        grad_outputs=grad_outputs,
    )
    torch.testing.assert_close(hidden_grad_compact, hidden_grad_ref, rtol=0, atol=0)
    torch.testing.assert_close(probs_grad_compact, probs_grad_ref, rtol=0, atol=0)
