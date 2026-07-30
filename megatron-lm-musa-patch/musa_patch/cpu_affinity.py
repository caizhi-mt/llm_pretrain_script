"""Optional per-local-rank CPU affinity for MUSA training workers."""

from __future__ import annotations

import os
import threading


_thread_state = threading.local()


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def parse_cpu_set(spec: str) -> set[int]:
    """Parse comma-separated CPU ids/ranges such as ``0-3,8,10-11``."""
    cpus: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid CPU range {item!r}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError(f"invalid CPU id {item!r}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU set must not be empty")
    return cpus


def maybe_bind_local_rank_cpu_affinity(stage: str = "early") -> set[int] | None:
    """Bind this worker according to ``MUSA_CPU_AFFINITY_MAP`` when enabled.

    The map contains one CPU set per local rank, separated by semicolons. For
    example: ``0-7;8-15;16-23;24-31``. The default ``mate`` mode binds only the
    thread submitting MATE work, after DeepEP has created its communication
    resources. The experimental ``early`` mode binds before torch is imported,
    so all subsequently created threads inherit the affinity.
    """
    if not _env_flag("MUSA_CPU_AFFINITY", "0"):
        return None

    mode = os.getenv("MUSA_CPU_AFFINITY_MODE", "mate")
    if mode not in {"early", "mate"}:
        raise ValueError(
            "MUSA_CPU_AFFINITY_MODE must be 'early' or 'mate', "
            f"got {mode!r}"
        )
    if mode != stage:
        return None

    affinity_map = os.getenv("MUSA_CPU_AFFINITY_MAP", "")
    entries = [entry.strip() for entry in affinity_map.split(";")]
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= len(entries) or not entries[local_rank]:
        raise ValueError(
            "MUSA_CPU_AFFINITY_MAP must provide a non-empty CPU set for "
            f"LOCAL_RANK={local_rank}, got {affinity_map!r}"
        )

    cache_key = (stage, local_rank, affinity_map)
    bound = getattr(_thread_state, "bound", {})
    if cache_key in bound:
        return set(bound[cache_key])

    requested = parse_cpu_set(entries[local_rank])
    available = set(os.sched_getaffinity(0))
    unavailable = requested - available
    if unavailable:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} requested CPUs outside its cpuset: "
            f"{sorted(unavailable)}"
        )

    os.sched_setaffinity(0, requested)
    actual = set(os.sched_getaffinity(0))
    if actual != requested:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} affinity mismatch: "
            f"requested={sorted(requested)}, actual={sorted(actual)}"
        )
    bound[cache_key] = frozenset(actual)
    _thread_state.bound = bound
    print(
        f"[MUSA_CPU_AFFINITY] mode={mode} local_rank={local_rank} "
        f"native_thread_id={threading.get_native_id()} cpus={sorted(actual)}",
        flush=True,
    )
    return actual
