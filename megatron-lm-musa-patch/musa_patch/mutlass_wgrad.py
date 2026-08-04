"""JIT loader and guarded dispatch for the MP31 MUTLASS MoE wgrad kernel."""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
from typing import Sequence

import torch


_SUPPORTED_SHAPES = {
    # (experts, out_features, in_features): maximum tested average tokens
    (128, 1536, 2048): 512,
    (128, 2048, 768): 512,
    (32, 4096, 7168): 2304,
}


def _aligned(tensor: torch.Tensor, alignment_bytes: int) -> bool:
    return tensor.data_ptr() % alignment_bytes == 0


def is_mp31_device(device: torch.device | int = 0) -> bool:
    if not torch.musa.is_available():
        return False
    properties = torch.musa.get_device_properties(device)
    return (properties.major, properties.minor) == (3, 1)


def select_mutlass_wgrad(
    *,
    num_experts: int,
    total_tokens: int,
    out_features: int,
    in_features: int,
    max_tokens: int,
    use_main_grad: bool,
    accumulate: bool,
) -> str | None:
    """Return the tuned kernel family, or None when TE is the safe choice."""
    if not use_main_grad or not accumulate:
        return None
    max_average = _SUPPORTED_SHAPES.get(
        (num_experts, out_features, in_features)
    )
    if max_average is None or total_tokens > num_experts * max_average:
        return None
    # The tested nonuniform distributions reach roughly 1.55x the average.
    # Keep an explicit tail guard so a single overloaded expert cannot select
    # a small-M tuning based only on the mean.
    if max_tokens > 2 * max_average:
        return None
    if num_experts == 32:
        return "e32_gate_up"
    return "down" if in_features <= 1024 else "gate_up"


@functools.lru_cache(maxsize=1)
def load_mutlass_wgrad():
    """Build/load the native extension once from MATE's bundled MUTLASS."""
    import mate
    import torch_musa
    from torch_musa.utils.musa_extension import load_inline

    module_dir = Path(__file__).resolve().parent
    source_path = module_dir / "csrc" / "mutlass_wgrad_kernel.mu"
    wrapper_path = module_dir / "csrc" / "mcc_wrapper.sh"
    source = source_path.read_text(encoding="utf-8")
    musa_flags = [
        "-O3",
        "-std=c++17",
        "--offload-arch=mp_31",
        "-fno-strict-aliasing",
        "-Od3",
        "-DMUTLASS_VERSIONS_GENERATED",
    ]
    build_identity = [
        source,
        wrapper_path.read_text(encoding="utf-8"),
        *musa_flags,
        f"mate={mate.__version__}",
        f"torch={torch.__version__}",
        f"torch_musa={torch_musa.__version__}",
    ]
    digest = hashlib.sha256("\0".join(build_identity).encode("utf-8")).hexdigest()[:12]

    mutlass_root = Path(mate.__file__).resolve().parent / "data" / "mutlass"
    include_paths = [
        mutlass_root / "include",
        mutlass_root / "tools" / "util" / "include",
    ]
    missing = [path for path in include_paths if not path.is_dir()]
    if missing:
        raise RuntimeError(f"MUTLASS include directories are missing: {missing}")

    os.environ.setdefault("PYTORCH_MCC", str(wrapper_path))
    os.environ.setdefault("MAX_JOBS", "8")
    declaration = r"""
#include <cstdint>
#include <vector>

void mutlass_wgrad_accumulate_musa(
    torch::Tensor input,
    torch::Tensor grad_output,
    std::vector<int64_t> counts,
    std::vector<torch::Tensor> outputs);
"""
    return load_inline(
        name=f"megatron_mutlass_wgrad_mp31_{digest}",
        cpp_sources=declaration,
        musa_sources=source,
        functions=["mutlass_wgrad_accumulate_musa"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_musa_cflags=musa_flags,
        extra_include_paths=[str(path) for path in include_paths],
        verbose=os.getenv("MUTLASS_WGRAD_BUILD_VERBOSE", "0") == "1",
        with_musa=True,
    )


def can_use_mutlass_wgrad(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    counts: Sequence[int],
    outputs: Sequence[torch.Tensor],
    *,
    use_main_grad: bool,
    accumulate: bool,
) -> str | None:
    """Validate dynamic tensors and select a benchmarked native kernel."""
    if not counts or len(counts) != len(outputs):
        return None
    if (
        input.device.type != "musa"
        or grad_output.device != input.device
        or input.dtype != torch.bfloat16
        or grad_output.dtype != torch.bfloat16
        or input.ndim != 2
        or grad_output.ndim != 2
        or input.shape[0] != grad_output.shape[0]
        or not input.is_contiguous()
        or not grad_output.is_contiguous()
        or not _aligned(input, 4)
        or not _aligned(grad_output, 4)
    ):
        return None
    if not is_mp31_device(input.device):
        return None
    if any(
        output.device != input.device
        or output.dtype != torch.float32
        or output.shape != (grad_output.shape[1], input.shape[1])
        or not output.is_contiguous()
        or not _aligned(output, 8)
        for output in outputs
    ):
        return None
    return select_mutlass_wgrad(
        num_experts=len(counts),
        total_tokens=input.shape[0],
        out_features=grad_output.shape[1],
        in_features=input.shape[1],
        max_tokens=max(counts),
        use_main_grad=use_main_grad,
        accumulate=accumulate,
    )


def mutlass_wgrad_accumulate(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    counts: Sequence[int],
    outputs: Sequence[torch.Tensor],
) -> None:
    """Accumulate packed expert wgrad directly into FP32 main_grad tensors."""
    extension = load_mutlass_wgrad()
    extension.mutlass_wgrad_accumulate_musa(
        input, grad_output, list(counts), list(outputs)
    )
