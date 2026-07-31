"""CPU-only tests for optional local-rank affinity binding."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "musa_patch" / "cpu_affinity.py"
SPEC = importlib.util.spec_from_file_location("cpu_affinity_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_cpu_set_supports_ranges_and_ids():
    assert MODULE.parse_cpu_set("0-2,5,7-8") == {0, 1, 2, 5, 7, 8}
    with pytest.raises(ValueError, match="invalid CPU range"):
        MODULE.parse_cpu_set("3-1")
    with pytest.raises(ValueError, match="must not be empty"):
        MODULE.parse_cpu_set("")


def test_affinity_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MUSA_CPU_AFFINITY", raising=False)
    assert MODULE.maybe_bind_local_rank_cpu_affinity() is None


def test_affinity_binds_the_local_rank_entry(monkeypatch):
    current = {0, 1, 2, 3}

    monkeypatch.setenv("MUSA_CPU_AFFINITY", "1")
    monkeypatch.setenv("MUSA_CPU_AFFINITY_MODE", "early")
    monkeypatch.setenv("MUSA_CPU_AFFINITY_MAP", "0-1;2-3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(MODULE.os, "sched_getaffinity", lambda _pid: current.copy())

    def set_affinity(_pid, cpus):
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(MODULE.os, "sched_setaffinity", set_affinity)
    assert MODULE.maybe_bind_local_rank_cpu_affinity() == {2, 3}


def test_mate_mode_skips_early_binding_and_binds_at_mate(monkeypatch):
    current = {0, 1, 2, 3}
    calls = []

    monkeypatch.setenv("MUSA_CPU_AFFINITY", "1")
    monkeypatch.setenv("MUSA_CPU_AFFINITY_MODE", "mate")
    monkeypatch.setenv("MUSA_CPU_AFFINITY_MAP", "0-1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(MODULE.os, "sched_getaffinity", lambda _pid: current.copy())

    def set_affinity(_pid, cpus):
        calls.append(set(cpus))
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(MODULE.os, "sched_setaffinity", set_affinity)
    assert MODULE.maybe_bind_local_rank_cpu_affinity("early") is None
    assert calls == []
    assert MODULE.maybe_bind_local_rank_cpu_affinity("mate") == {0, 1}
    assert calls == [{0, 1}]


def test_affinity_rejects_cpus_outside_cpuset(monkeypatch):
    monkeypatch.setenv("MUSA_CPU_AFFINITY", "1")
    monkeypatch.setenv("MUSA_CPU_AFFINITY_MODE", "early")
    monkeypatch.setenv("MUSA_CPU_AFFINITY_MAP", "0-7")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(MODULE.os, "sched_getaffinity", lambda _pid: {0, 1})
    with pytest.raises(RuntimeError, match="outside its cpuset"):
        MODULE.maybe_bind_local_rank_cpu_affinity()
