"""Optional GM6 TN kernel for TE BF16 grouped wgrad on MUSA.

This patches Transformer Engine's ``general_grouped_gemm`` entry point, not
the optional MATE fprop/dgrad path.  It is disabled unless
``TE_TN_GM6_WGRAD=1`` is set.
"""

from __future__ import annotations

import functools
import os
from numbers import Integral
from typing import Sequence

import torch


def _same_packed_storage(tensors: Sequence[torch.Tensor]) -> bool:
    if not tensors:
        return False
    first = tensors[0]
    if not isinstance(first, torch.Tensor) or not first.is_contiguous():
        return False
    storage = first.untyped_storage().data_ptr()
    next_ptr = first.data_ptr()
    for tensor in tensors:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dim() != 2
            or not tensor.is_contiguous()
            or tensor.dtype != first.dtype
            or tensor.device != first.device
            or tensor.untyped_storage().data_ptr() != storage
            or tensor.data_ptr() != next_ptr
        ):
            return False
        next_ptr += tensor.numel() * tensor.element_size()
    return True


def _eligible(
    A,
    B,
    out,
    layout,
    m_splits,
    gelu,
    grad,
    accumulate,
    bias,
    use_bias,
    D_dtype,
    single_output,
) -> bool:
    del accumulate, bias
    if os.getenv("ENABLE_ZERO_BUBBLE", "0") == "1":
        return False
    if layout != "NT" or not grad or gelu or use_bias or single_output or D_dtype is not None:
        return False
    if m_splits is None or len(A) <= 1 or len(A) != len(B) or len(A) != len(out):
        return False
    if len(m_splits) != len(A) or not all(isinstance(size, Integral) and size >= 0 for size in m_splits):
        return False
    if not any(m_splits):
        return False
    active = [index for index, size in enumerate(m_splits) if size]
    if not (_same_packed_storage([A[index] for index in active]) and
            _same_packed_storage([B[index] for index in active])):
        return False
    if A[active[0]].dtype != torch.bfloat16 or B[active[0]].dtype != torch.bfloat16:
        return False
    if out[active[0]].dtype != torch.float32 or not getattr(A[active[0]], "is_musa", False):
        return False
    in_features = A[active[0]].shape[1]
    out_features = B[active[0]].shape[1]
    for index, size in enumerate(m_splits):
        if size and (A[index].shape != (size, in_features) or B[index].shape != (size, out_features)):
            return False
        if (out[index].shape != (out_features, in_features) or
                out[index].dtype != torch.float32 or
                out[index].device != out[active[0]].device):
            return False
    return True


def _active_runs(sizes: Sequence[int]) -> list[tuple[int, int]]:
    """Return [start, end) runs; an empty expert is never passed to GM6."""
    runs: list[tuple[int, int]] = []
    start = None
    for index, size in enumerate(sizes):
        if size and start is None:
            start = index
        elif not size and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(sizes)))
    return runs


@functools.lru_cache(maxsize=64)
def _group_sizes(sizes: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(sizes, dtype=torch.int64, device="cpu")


def install_te_tn_gm6() -> None:
    """Install the opt-in wrapper on every TE Python binding of the function."""
    import transformer_engine.pytorch.cpp_extensions as cpp_extensions
    import transformer_engine.pytorch.cpp_extensions.gemm as gemm_module
    import transformer_engine.pytorch.module.grouped_linear as grouped_linear

    if getattr(gemm_module.general_grouped_gemm, "_te_tn_gm6_wrapper", False):
        return

    original = gemm_module.general_grouped_gemm
    logged = False

    @functools.wraps(original)
    def wrapped(
        A,
        B,
        out,
        out_dtype,
        workspaces,
        layout="TN",
        m_splits=None,
        gelu=False,
        grad=False,
        accumulate=False,
        bias=None,
        use_bias=False,
        use_split_accumulator=False,
        D_dtype=None,
        single_output=False,
    ):
        nonlocal logged
        if not _eligible(
            A, B, out, layout, m_splits, gelu, grad, accumulate,
            bias, use_bias, D_dtype, single_output,
        ):
            return original(
                A, B, out, out_dtype, workspaces, layout, m_splits, gelu,
                grad, accumulate, bias, use_bias, use_split_accumulator,
                D_dtype, single_output,
            )

        from musa_patch.tn_gm6 import grouped_wgrad_bf16_fp32
        from transformer_engine.pytorch.cpp_extensions.gemm import _empty_tensor

        sizes = tuple(int(size) for size in m_splits)
        active_runs = _active_runs(sizes)
        in_features = A[next(index for index, size in enumerate(sizes) if size)].shape[1]
        out_features = B[next(index for index, size in enumerate(sizes) if size)].shape[1]
        if not accumulate:
            for index, size in enumerate(sizes):
                if not size:
                    # The original explicit path allocates a zero output for
                    # an empty expert.  With main_grad accumulation, beta=1
                    # must instead leave that expert untouched.
                    out[index].zero_()
        if not logged:
            rank = int(os.getenv("RANK", "0"))
            if rank == 0:
                print(
                    "[TE_TN_GM6] using BF16xBF16->FP32 grouped NT wgrad "
                    f"groups={len(sizes)} active_runs={active_runs} in={in_features} out={out_features}",
                    flush=True,
                )
            logged = True
        for start, end in active_runs:
            run_sizes = sizes[start:end]
            if len(run_sizes) < 2 or not _same_packed_storage(out[start:end]):
                original(
                    list(A[start:end]), list(B[start:end]), list(out[start:end]),
                    out_dtype, workspaces, "NT", list(run_sizes), False, True,
                    accumulate, bias, use_bias, use_split_accumulator, D_dtype,
                    single_output,
                )
                continue
            total_rows = sum(run_sizes)
            packed_input = A[start].as_strided((total_rows, in_features), (in_features, 1))
            packed_grad_output = B[start].as_strided((total_rows, out_features), (out_features, 1))
            packed_output = out[start].as_strided(
                (end - start, out_features, in_features),
                (out_features * in_features, in_features, 1),
            )
            with torch.profiler.record_function("te_tn_gm6_wgrad"):
                grouped_wgrad_bf16_fp32(
                    packed_grad_output,
                    packed_input,
                    _group_sizes(run_sizes),
                    packed_output,
                    bool(accumulate),
                )
        empty = _empty_tensor()
        empty_tensors = [empty] * len(out)
        return out, empty_tensors, empty_tensors

    wrapped._te_tn_gm6_wrapper = True
    wrapped._te_tn_gm6_original = original
    gemm_module.general_grouped_gemm = wrapped
    cpp_extensions.general_grouped_gemm = wrapped
    grouped_linear.general_grouped_gemm = wrapped
