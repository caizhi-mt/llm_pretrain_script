"""Loader for the prebuilt standalone muDNN GM6 TN grouped-GEMM extension."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module


@lru_cache(maxsize=1)
def _extension():
    try:
        return import_module("musa_patch.tn_gm6._tn_gm6")
    except ImportError as exc:
        raise RuntimeError(
            "GM6 TN wgrad extension is not built. Run setup.py build_ext "
            "--inplace on a MUSA machine before enabling TE_TN_GM6_WGRAD."
        ) from exc


def preload_tn_gm6():
    """Load GM6 before Transformer Engine registers overlapping ASM symbols."""
    return _extension()


def grouped_wgrad_bf16_fp32(
    grad_output,
    inp,
    group_sizes_cpu,
    output,
    accumulate: bool,
):
    """Compute per-expert ``grad_output.T @ inp`` into packed FP32 output."""
    return _extension().grouped_wgrad_bf16_fp32(
        grad_output,
        inp,
        group_sizes_cpu,
        output,
        accumulate,
    )
