"""Focused CPU tests for the optional MATE GroupedLinear patch helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "musa_patch" / "mate_grouped_gemm.py"
SPEC = importlib.util.spec_from_file_location("mate_grouped_gemm_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_env_flag_accepts_only_zero_or_one(monkeypatch):
    monkeypatch.delenv("MATE_TEST_FLAG", raising=False)
    assert MODULE.env_flag("MATE_TEST_FLAG", "0") is False
    assert MODULE.env_flag("MATE_TEST_FLAG", "1") is True

    monkeypatch.setenv("MATE_TEST_FLAG", "2")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        MODULE.env_flag("MATE_TEST_FLAG")


def test_host_splits_reuses_attached_deepep_metadata():
    counts = torch.tensor([2, 3], dtype=torch.int32)
    counts._mate_m_splits = (2, 3)
    assert MODULE._host_splits(counts) == [2, 3]


def test_is_packed_checks_physical_weight_adjacency():
    packed = torch.empty((3, 4, 8), dtype=torch.bfloat16)
    assert MODULE._is_packed(list(packed.unbind(0)))

    unpacked = [torch.empty((4, 8), dtype=torch.bfloat16) for _ in range(3)]
    assert not MODULE._is_packed(unpacked)


def test_mubin_dispatch_cache_reuses_metadata_but_not_custom_repository(tmp_path):
    calls = {"dispatcher": 0, "kernel": 0}

    def dispatcher(path):
        calls["dispatcher"] += 1
        return object()

    def ensure_kernel(module, module_dir, kernel_name, repository=None):
        calls["kernel"] += 1
        return Path(module_dir) / f"{kernel_name}.mu"

    gemm_api = SimpleNamespace(MoeGemmMubinDispatcher=dispatcher)
    dispatch = SimpleNamespace(ensure_mubin_kernel_artifact=ensure_kernel)
    assert MODULE._install_mubin_dispatch_cache(gemm_api, dispatch)
    assert not MODULE._install_mubin_dispatch_cache(gemm_api, dispatch)

    first = gemm_api.MoeGemmMubinDispatcher(tmp_path / "kernel_map.json")
    second = gemm_api.MoeGemmMubinDispatcher(str(tmp_path / "kernel_map.json"))
    assert first is second
    assert calls["dispatcher"] == 1

    for _ in range(2):
        dispatch.ensure_mubin_kernel_artifact("gemm", tmp_path, "kernel_a")
    assert calls["kernel"] == 1

    repository = object()
    for _ in range(2):
        dispatch.ensure_mubin_kernel_artifact(
            "gemm", tmp_path, "kernel_a", repository=repository
        )
    assert calls["kernel"] == 3
