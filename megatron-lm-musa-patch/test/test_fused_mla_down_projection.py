"""Tests for the checkpoint-compatible MLA q/kv down-projection fusion."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "musa_patch" / "fused_mla_down_projection.py"
SPEC = importlib.util.spec_from_file_location(
    "fused_mla_down_projection_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _config(**kwargs):
    values = {
        "q_lora_rank": 512,
        "tensor_model_parallel_size": 1,
        "add_bias_linear": False,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_feature_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("MUSA_FUSED_MLA_DOWN_PROJ", raising=False)
    assert MODULE._env_flag("MUSA_FUSED_MLA_DOWN_PROJ")

    monkeypatch.setenv("MUSA_FUSED_MLA_DOWN_PROJ", "0")
    assert not MODULE._env_flag("MUSA_FUSED_MLA_DOWN_PROJ")


def test_cpu_and_unsupported_configs_fall_back(monkeypatch):
    monkeypatch.setenv("MUSA_FUSED_MLA_DOWN_PROJ", "1")
    hidden = torch.empty((8, 2048), dtype=torch.bfloat16)
    q_module = torch.nn.Linear(2048, 512, bias=False, dtype=torch.bfloat16)
    kv_module = torch.nn.Linear(2048, 192, bias=False, dtype=torch.bfloat16)

    assert not MODULE.is_supported(hidden, q_module, kv_module, _config())
    assert not MODULE.is_supported(
        hidden, q_module, kv_module, _config(tensor_model_parallel_size=2)
    )
    assert not MODULE.is_supported(
        hidden, q_module, kv_module, _config(add_bias_linear=True)
    )


@pytest.mark.skipif(
    os.getenv("RUN_MUSA_TESTS", "0") != "1"
    or not hasattr(torch, "musa")
    or not torch.musa.is_available(),
    reason="set RUN_MUSA_TESTS=1 on a MUSA host",
)
def test_musa_forward_dgrad_and_main_grad_match_reference():
    torch.manual_seed(1234)
    tokens, hidden_size, q_rank, kv_rank = 4096, 2048, 512, 192
    hidden = torch.randn(
        tokens,
        hidden_size,
        device="musa",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    q_weight = torch.nn.Parameter(
        torch.randn(q_rank, hidden_size, device="musa", dtype=torch.bfloat16)
    )
    kv_weight = torch.nn.Parameter(
        torch.randn(kv_rank, hidden_size, device="musa", dtype=torch.bfloat16)
    )
    for weight in (q_weight, kv_weight):
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        weight.grad_added_to_main_grad = False
        weight.zero_out_wgrad = False

    q_output, kv_output = MODULE._FusedMLADownProjection.apply(
        hidden, q_weight, kv_weight
    )
    q_reference = torch.nn.functional.linear(hidden, q_weight)
    kv_reference = torch.nn.functional.linear(hidden, kv_weight)
    torch.testing.assert_close(q_output, q_reference, rtol=0, atol=0)
    torch.testing.assert_close(kv_output, kv_reference, rtol=0, atol=0)

    q_grad = torch.randn_like(q_output)
    kv_grad = torch.randn_like(kv_output)
    hidden_grad, q_grad_result, kv_grad_result = torch.autograd.grad(
        (q_output, kv_output),
        (hidden, q_weight, kv_weight),
        grad_outputs=(q_grad, kv_grad),
        allow_unused=True,
    )
    hidden_reference = torch.matmul(q_grad, q_weight) + torch.matmul(kv_grad, kv_weight)
    q_wgrad_reference = torch.mm(q_grad.T.float(), hidden.detach().float())
    kv_wgrad_reference = torch.mm(kv_grad.T.float(), hidden.detach().float())

    assert q_grad_result is None
    assert kv_grad_result is None
    assert q_weight.grad_added_to_main_grad
    assert kv_weight.grad_added_to_main_grad
    torch.testing.assert_close(hidden_grad, hidden_reference, rtol=2e-2, atol=0.25)
    torch.testing.assert_close(
        q_weight.main_grad, q_wgrad_reference, rtol=2e-2, atol=0.5
    )
    torch.testing.assert_close(
        kv_weight.main_grad, kv_wgrad_reference, rtol=2e-2, atol=0.5
    )

    hidden_second = hidden.detach().clone().requires_grad_(True)
    q_second, kv_second = MODULE._FusedMLADownProjection.apply(
        hidden_second, q_weight, kv_weight
    )
    torch.autograd.grad(
        (q_second, kv_second),
        (hidden_second, q_weight, kv_weight),
        grad_outputs=(q_grad, kv_grad),
        allow_unused=True,
    )
    torch.testing.assert_close(
        q_weight.main_grad, 2 * q_wgrad_reference, rtol=2e-2, atol=1.0
    )
    torch.testing.assert_close(
        kv_weight.main_grad, 2 * kv_wgrad_reference, rtol=2e-2, atol=1.0
    )
