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


def test_mubin_dispatch_cache_reuses_resolution_and_default_metadata(tmp_path):
    calls = {"dispatcher": 0, "kernel": 0, "module": 0, "resolve": 0}

    class Dispatcher:
        def __init__(self, path):
            calls["dispatcher"] += 1
            self.path = Path(path)

        def resolve_kernel_path(self, args, get_asm_id, mubin_dir):
            calls["resolve"] += 1
            asm_id = get_asm_id(args)
            return Path(mubin_dir) / f"kernel_{asm_id}.mu"

    def ensure_kernel(module, module_dir, kernel_name, repository=None):
        calls["kernel"] += 1
        return Path(module_dir) / f"{kernel_name}.mu"

    def ensure_module(module, cache_dir=None, repository=None):
        calls["module"] += 1
        root = Path(cache_dir) if cache_dir is not None else tmp_path
        return root / str(module)

    gemm_api = SimpleNamespace(
        MoeGemmMubinDispatcher=Dispatcher,
        ensure_mubin_module_artifacts=ensure_module,
    )
    dispatch = SimpleNamespace(ensure_mubin_kernel_artifact=ensure_kernel)
    assert MODULE._install_mubin_dispatch_cache(gemm_api, dispatch)
    assert not MODULE._install_mubin_dispatch_cache(gemm_api, dispatch)

    first = gemm_api.MoeGemmMubinDispatcher(tmp_path / "kernel_map.json")
    second = gemm_api.MoeGemmMubinDispatcher(str(tmp_path / "kernel_map.json"))
    assert first is second
    assert calls["dispatcher"] == 1

    get_asm_id = lambda args: args.asm_id
    first_path = first.resolve_kernel_path(
        SimpleNamespace(asm_id=("bf16", "K", "K")),
        get_asm_id,
        tmp_path / "mubin",
    )
    second_path = first.resolve_kernel_path(
        SimpleNamespace(asm_id=("bf16", "K", "K"), dynamic_m=1234),
        get_asm_id,
        str(tmp_path / "mubin"),
    )
    assert first_path == second_path
    assert calls["resolve"] == 1

    first.resolve_kernel_path(
        SimpleNamespace(asm_id=("bf16", "K", "N")),
        get_asm_id,
        tmp_path / "mubin",
    )
    assert calls["resolve"] == 2

    for _ in range(2):
        gemm_api.ensure_mubin_module_artifacts("gemm")
    assert calls["module"] == 1

    gemm_api.ensure_mubin_module_artifacts("gemm", cache_dir=tmp_path / "custom")
    gemm_api.ensure_mubin_module_artifacts("gemm", cache_dir=tmp_path / "custom")
    assert calls["module"] == 3

    for _ in range(2):
        dispatch.ensure_mubin_kernel_artifact("gemm", tmp_path, "kernel_a")
    assert calls["kernel"] == 1

    repository = object()
    for _ in range(2):
        dispatch.ensure_mubin_kernel_artifact(
            "gemm", tmp_path, "kernel_a", repository=repository
        )
    assert calls["kernel"] == 3


def test_static_layout_support_caches_only_successful_checks(monkeypatch):
    module = SimpleNamespace(num_gemms=2)
    packed = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    module.weight0, module.weight1 = packed.unbind(0)
    module.weight0.main_grad = torch.empty((4, 8), dtype=torch.float32)
    module.weight1.main_grad = torch.empty((4, 8), dtype=torch.float32)

    calls = {"packed": 0}
    original_is_packed = MODULE._is_packed

    def counted_is_packed(weights):
        calls["packed"] += 1
        return original_is_packed(weights)

    monkeypatch.setattr(MODULE, "_is_packed", counted_is_packed)
    assert MODULE._static_layout_supported(module, torch.device("cpu"), True)
    assert MODULE._static_layout_supported(module, torch.device("cpu"), True)
    assert calls["packed"] == 1
    assert all(isinstance(value, bool) for value in module._mate_static_support_cache.values())

    not_ready = SimpleNamespace(num_gemms=1)
    not_ready.weight0 = torch.empty((4, 8), dtype=torch.bfloat16)
    assert not MODULE._static_layout_supported(not_ready, torch.device("cpu"), True)
    not_ready.weight0.main_grad = torch.empty((4, 8), dtype=torch.float32)
    assert MODULE._static_layout_supported(not_ready, torch.device("cpu"), True)
