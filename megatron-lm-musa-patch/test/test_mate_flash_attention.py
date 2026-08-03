"""Focused CPU tests for the optional MATE FlashAttention patch helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "musa_patch" / "mate_flash_attention.py"
SPEC = importlib.util.spec_from_file_location(
    "mate_flash_attention_test_module", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_env_flag_accepts_only_zero_or_one(monkeypatch):
    monkeypatch.delenv("MATE_TEST_FLAG", raising=False)
    assert MODULE.env_flag("MATE_TEST_FLAG", "0") is False
    assert MODULE.env_flag("MATE_TEST_FLAG", "1") is True

    monkeypatch.setenv("MATE_TEST_FLAG", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        MODULE.env_flag("MATE_TEST_FLAG")


def test_launch_cache_keys_all_result_affecting_metadata(monkeypatch):
    calls = []

    def resolve(dtype, causal, qk_dim, v_dim):
        calls.append((dtype, causal, qk_dim, v_dim))
        return object(), "kernel.mu", v_dim

    monkeypatch.setattr(MODULE, "_resolve_mubin_launch", resolve)
    MODULE._cached_mubin_launch.cache_clear()
    key = ("0.2.5", "/mubin", 0, (3, 1), torch.bfloat16, True, 192, 128)

    first = MODULE._cached_mubin_launch(*key)
    second = MODULE._cached_mubin_launch(*key)
    third = MODULE._cached_mubin_launch(*key[:-1], 64)

    assert first is second
    assert third is not first
    assert calls == [
        (torch.bfloat16, True, 192, 128),
        (torch.bfloat16, True, 192, 64),
    ]


def test_cpu_or_unvalidated_shape_uses_original_function():
    calls = []
    sentinel = object()

    def original(*args):
        calls.append(args)
        return sentinel

    patched = MODULE._make_patched_flash_attn_func(original)
    q = torch.empty((2, 16, 4, 192), dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty((2, 16, 4, 128), dtype=torch.bfloat16)

    assert patched(q, k, v, causal=True) is sentinel
    assert len(calls) == 1


@pytest.mark.parametrize("window", [None, (-1, -1), (-1, 0)])
def test_full_causal_window_forms_are_accepted(window):
    assert MODULE._is_full_causal_window(window)


@pytest.mark.parametrize("window", [(0, 0), (128, 0), (-1, 128)])
def test_sliding_window_forms_are_rejected(window):
    assert not MODULE._is_full_causal_window(window)


def test_fixed_length_launch_passes_no_varlen_metadata():
    calls = []

    def launch(*args):
        calls.append(args)

    values = tuple(object() for _ in range(6))
    MODULE._launch_fixed_length_mubin(launch, *values)

    assert len(calls) == 1
    assert calls[0][:6] == values
    assert calls[0][6:] == (None, None, None, None)
