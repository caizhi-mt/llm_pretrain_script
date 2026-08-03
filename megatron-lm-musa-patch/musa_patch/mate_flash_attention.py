"""MATE MLA FlashAttention forward with the native MUSA backward.

MATE 0.2.5 has a faster MUBIN forward kernel for the DeepSeek MLA
``Dqk=192, Dv=128`` shape, but its varlen fallback backward is both slower
and numerically unsuitable for training.  This patch therefore keeps the
existing MUSA SDPA backward and only replaces the forward.  Immutable MUBIN
selection and the loaded launch function are cached; no tensor, sequence
content, or attention result is cached.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, Optional, Tuple

import torch


_ORIGINAL_FLASH_ATTN_FUNC: Optional[Callable] = None
_ACTIVE_SHAPES: set[tuple] = set()


def env_flag(name: str, default: str = "0") -> bool:
    """Read a strict 0/1 environment flag."""
    value = os.getenv(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.getenv("RANK", "0"))


def _log(message: str) -> None:
    if _rank() == 0:
        print(f"[MATE_FLASH_ATTN] {message}", flush=True)


@functools.lru_cache(maxsize=1)
def load_mate_flash_attention():
    """Import MATE after torch_musa has registered the MUSA tensor bridge."""
    import torch_musa  # noqa: F401
    import mate

    return mate


def _device_index(tensor: torch.Tensor) -> int:
    if tensor.device.index is not None:
        return tensor.device.index
    return torch.musa.current_device()


@functools.lru_cache(maxsize=None)
def _empty_dropout_mask(device_index: int) -> torch.Tensor:
    """Return the immutable empty mask expected by dropout-free native bwd."""
    with torch.musa.device(device_index):
        return torch.empty((0,), dtype=torch.bool, device=f"musa:{device_index}")


def _context_parallel_world_size() -> int:
    """Read CP after Megatron process groups have been initialized."""
    try:
        from megatron.core import parallel_state

        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_world_size()
    except (AssertionError, ImportError):
        pass
    return 1


def _is_full_causal_window(window_size) -> bool:
    if window_size is None:
        window_size = (-1, -1)
    return tuple(window_size) in {(-1, -1), (-1, 0)}


def _support_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    causal: bool,
    window_size,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    return_attn_probs: bool,
) -> Optional[str]:
    """Return ``None`` only for the validated DeepSeek MLA training path."""
    if os.getenv("USE_RECOMPUTE_VARIANCE", "0") == "1":
        return "custom recompute-variance attention is not supported"
    if _context_parallel_world_size() != 1:
        return "context parallel attention is not supported"
    if not all(isinstance(x, torch.Tensor) for x in (q, k, v)):
        return "q/k/v must be tensors"
    if not all(bool(getattr(x, "is_musa", False)) for x in (q, k, v)):
        return "q/k/v must be MUSA tensors"
    if q.device != k.device or q.device != v.device:
        return "q/k/v must be on the same device"
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return "only BF16 q/k/v are validated"
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return "only fixed-length BSHD tensors are supported"
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return "the last q/k/v dimension must be contiguous"

    batch, seqlen, num_heads, qk_dim = q.shape
    if tuple(k.shape) != (batch, seqlen, num_heads, qk_dim):
        return "current MLA path requires matching q/k shapes"
    if tuple(v.shape) != (batch, seqlen, num_heads, 128):
        return "current MLA path requires V head_dim=128 and matching B/S/H"
    if batch <= 0 or seqlen <= 0 or num_heads <= 0 or qk_dim != 192:
        return "current MLA path requires non-empty Dqk=192 tensors"
    if not causal:
        return "only causal attention is validated"
    if float(dropout_p) != 0.0:
        return "attention dropout is not supported"
    # TE normalizes full causal attention to (-1, 0); FlashAttention's public
    # default is (-1, -1). They are equivalent for this causal fixed-length path.
    if not _is_full_causal_window(window_size):
        return "sliding-window attention is not supported"
    if float(softcap) != 0.0:
        return "softcap is not supported"
    if alibi_slopes is not None:
        return "ALiBi is not supported"
    if deterministic:
        return "deterministic mode is not supported"
    if return_attn_probs:
        return "return_attn_probs is not supported"
    if tuple(torch.musa.get_device_capability(_device_index(q))) != (3, 1):
        return "MATE MUBIN fast path requires MP31"
    return None


def _resolve_mubin_launch(
    dtype: torch.dtype,
    causal: bool,
    qk_dim: int,
    v_dim: int,
):
    """Resolve and load one MATE MUBIN launch function."""
    from mate.artifacts import ensure_mubin_module_artifacts
    from mate.jit.mubin.flash_attention.dispatch import FlashAttentionMubinDispatcher
    from mate.jit.mubin.flash_attention.flash_attention_api import _load_launch_function

    module_dir = ensure_mubin_module_artifacts("flash_attention")
    dispatcher = FlashAttentionMubinDispatcher(module_dir / "kernel_map.json")
    asm_id = dispatcher.get_asm_id(dtype, causal, False, qk_dim)
    kernel_path = dispatcher.resolve_kernel_path(asm_id, module_dir / "mubin")
    launch = _load_launch_function(kernel_path.stem, str(kernel_path))
    return launch, str(kernel_path), v_dim


@functools.lru_cache(maxsize=None)
def _cached_mubin_launch(
    mate_version: str,
    mubin_dir: str,
    device_index: int,
    capability: Tuple[int, int],
    dtype: torch.dtype,
    causal: bool,
    qk_dim: int,
    v_dim: int,
):
    """Cache immutable artifact selection and the loaded launch handle."""
    del mate_version, mubin_dir, device_index, capability
    return _resolve_mubin_launch(dtype, causal, qk_dim, v_dim)


def _mate_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
):
    mate = load_mate_flash_attention()
    if not env_flag("MATE_CACHE_MUBIN_DISPATCH", "1"):
        return mate.flash_attn_varlen_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            backend="mubin",
            return_softmax_lse=True,
        )

    device_index = _device_index(q)
    capability = tuple(torch.musa.get_device_capability(device_index))
    launch, _, _ = _cached_mubin_launch(
        getattr(mate, "__version__", "unknown"),
        os.getenv("MATE_MUBIN_DIR", ""),
        device_index,
        capability,
        q.dtype,
        causal,
        q.shape[-1],
        v.shape[-1],
    )
    out = torch.empty(
        (*q.shape[:-1], v.shape[-1]),
        dtype=q.dtype,
        device=q.device,
    )
    softmax_lse = torch.empty(
        (q.shape[0], q.shape[2], q.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )
    _launch_fixed_length_mubin(
        launch,
        q,
        k,
        v,
        softmax_scale,
        out,
        softmax_lse,
    )
    return out, softmax_lse


def _launch_fixed_length_mubin(
    launch,
    q,
    k,
    v,
    softmax_scale,
    out,
    softmax_lse,
) -> None:
    """Call the MATE 0.2.5 non-varlen MUBIN ABI.

    The last four values are varlen-only metadata. MATE's own non-varlen API
    passes ``None`` for them and derives both sequence lengths from the 4-D
    Q/K tensors.
    """
    launch(
        q,
        k,
        v,
        softmax_scale,
        out,
        softmax_lse,
        None,
        None,
        None,
        None,
    )


class _MateFlashAttention(torch.autograd.Function):
    """MATE MUBIN forward paired with the existing MUSA SDPA backward."""

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale: float, causal: bool):
        with torch.profiler.record_function("mate_flash_attn_fwd"):
            out, softmax_lse = _mate_forward(q, k, v, softmax_scale, causal)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        # MATE consumes BSHD while the native MUSA SDPA op consumes BHSD.
        with torch.profiler.record_function("mate_flash_attn_native_bwd"):
            (
                dq,
                dk,
                dv,
                _,
            ) = torch.ops.aten._scaled_dot_product_attention_flash_musa_backward(
                grad_out.transpose(1, 2),
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                out.transpose(1, 2),
                softmax_lse,
                _empty_dropout_mask(_device_index(q)),
                ctx.causal,
                None,
                scale=ctx.softmax_scale,
            )
        if not ctx.needs_input_grad[0]:
            dq = None
        else:
            dq = dq.transpose(1, 2)
        if not ctx.needs_input_grad[1]:
            dk = None
        else:
            dk = dk.transpose(1, 2)
        if not ctx.needs_input_grad[2]:
            dv = None
        else:
            dv = dv.transpose(1, 2)
        return dq, dk, dv, None, None


def _make_patched_flash_attn_func(original_flash_attn_func: Callable) -> Callable:
    @functools.wraps(original_flash_attn_func)
    def patched_flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ):
        reason = _support_reason(
            q,
            k,
            v,
            dropout_p,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            return_attn_probs,
        )
        if reason is not None:
            return original_flash_attn_func(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                softcap,
                alibi_slopes,
                deterministic,
                return_attn_probs,
            )

        scale = (
            float(softmax_scale) if softmax_scale is not None else q.shape[-1] ** -0.5
        )
        # Once all ranks select the fast path, fail fast on a runtime/kernel
        # error instead of letting individual ranks silently diverge to native.
        out = _MateFlashAttention.apply(q, k, v, scale, causal)

        shape_key = (tuple(q.shape), tuple(v.shape), q.dtype, q.device)
        if shape_key not in _ACTIVE_SHAPES:
            _ACTIVE_SHAPES.add(shape_key)
            _log(
                f"active q={list(q.shape)} v={list(v.shape)} "
                f"cache={int(env_flag('MATE_CACHE_MUBIN_DISPATCH', '1'))} "
                "backward=native_musa"
            )
        return out

    return patched_flash_attn_func


def install_mate_flash_attention() -> None:
    """Patch Transformer Engine's FlashAttention entry point once."""
    global _ORIGINAL_FLASH_ATTN_FUNC

    import transformer_engine.pytorch.attention as te_attention

    mate = load_mate_flash_attention()
    from importlib.metadata import PackageNotFoundError, version

    try:
        mate_mubin_version = version("mate-mubin")
    except PackageNotFoundError:
        _log("mate-mubin is not installed; keeping native path")
        return
    mate_version = getattr(mate, "__version__", "unknown")
    if mate_version != "0.2.5" or mate_mubin_version != mate_version:
        _log(
            "validated versions are mate=mate-mubin=0.2.5; "
            f"got mate={mate_version} mate-mubin={mate_mubin_version}; keeping native path"
        )
        return

    if getattr(te_attention, "_mate_flash_attention_installed", False):
        return
    original = getattr(te_attention, "flash_attn_func", None)
    if original is None:
        _log(
            "Transformer Engine FlashAttention entry point unavailable; keeping native path"
        )
        return
    _ORIGINAL_FLASH_ATTN_FUNC = original
    te_attention.flash_attn_func = _make_patched_flash_attn_func(original)
    te_attention._mate_flash_attention_installed = True
    _log("installed: cached MATE MUBIN forward + native MUSA backward")
