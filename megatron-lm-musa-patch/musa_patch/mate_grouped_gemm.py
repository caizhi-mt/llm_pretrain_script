"""Opt-in MATE BF16 GroupedLinear fast path for MoE experts on MUSA.

The patch keeps Transformer Engine's module, parameters, and state-dict
format. MATE handles BF16 fprop and dgrad, while Transformer Engine handles
wgrad through one ``general_grouped_gemm`` call. The fast configuration writes
wgrad directly into the persistent FP32 ``main_grad`` buffer, avoiding a BF16
temporary gradient and the subsequent BF16-to-FP32 accumulation pass.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Sequence

import torch


def env_flag(name: str, default: str = "0") -> bool:
    """Read a strict 0/1 environment flag."""
    value = os.getenv(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


@functools.lru_cache(maxsize=1)
def load_mate_gemm():
    """Load MATE after torch_musa has registered its DLPack bridge."""
    import torch_musa  # noqa: F401
    from mate import gemm as mate_gemm

    if env_flag("MATE_CACHE_MUBIN_DISPATCH", "1"):
        try:
            from mate.jit.mubin.gemm import dispatch, gemm_api

            if _install_mubin_dispatch_cache(gemm_api, dispatch):
                _log("MUBIN dispatch cache installed")
        except (AttributeError, ImportError) as exc:
            # MATE's private MUBIN API may change between releases. Keep the
            # grouped GEMM path usable and make the missing optimization clear.
            _log(f"MUBIN dispatch cache unavailable: {exc}")

    return mate_gemm


def _install_mubin_dispatch_cache(gemm_api, dispatch) -> bool:
    """Cache immutable MATE MUBIN dispatch state, never tensors or routing counts.

    MATE 0.2.5 rebuilds ``MoeGemmMubinDispatcher`` and re-verifies the selected
    kernel artifact for every ragged-M call. It also recomputes the MUBIN id
    hash and resolves the same module directory and kernel path on every call.

    The selected ``GemmMubinId`` is immutable and already contains every field
    that changes the kernel variant, including dtype, layout, MP architecture,
    and the block selected from the current ragged-M shape. Cache by that id,
    while continuing to pass the current M and routing counts to every launch.
    """
    if getattr(gemm_api, "_megatron_mubin_dispatch_cache_installed", False):
        return False

    original_dispatcher = gemm_api.MoeGemmMubinDispatcher
    original_ensure_kernel = dispatch.ensure_mubin_kernel_artifact
    original_ensure_module = getattr(gemm_api, "ensure_mubin_module_artifacts", None)

    if callable(original_ensure_module):

        @functools.lru_cache(maxsize=None)
        def _default_module_dir(module: str):
            return original_ensure_module(module)

        def cached_ensure_module(module, cache_dir=None, repository=None):
            # The packaged training path always uses the immutable default
            # artifact location. Preserve MATE's behavior for explicit or
            # potentially mutable repositories.
            if cache_dir is not None or repository is not None:
                return original_ensure_module(
                    module, cache_dir=cache_dir, repository=repository
                )
            return _default_module_dir(str(module))

        gemm_api.ensure_mubin_module_artifacts = cached_ensure_module

    @functools.lru_cache(maxsize=None)
    def _dispatcher_for_path(kernel_map_path: str):
        dispatcher_instance = original_dispatcher(Path(kernel_map_path))
        original_resolve_kernel_path = getattr(
            dispatcher_instance, "resolve_kernel_path", None
        )
        if not callable(original_resolve_kernel_path):
            return dispatcher_instance

        @functools.lru_cache(maxsize=None)
        def _resolved_kernel_path(asm_id, mubin_dir: str):
            # MATE's resolver only uses ``args`` through ``get_asm_id(args)``.
            # Supplying the already-computed immutable id keeps its lookup,
            # validation, and error behavior intact on a cache miss.
            return original_resolve_kernel_path(
                None,
                lambda _unused_args: asm_id,
                Path(mubin_dir),
            )

        def cached_resolve_kernel_path(args, get_asm_id, mubin_dir):
            asm_id = get_asm_id(args)
            try:
                hash(asm_id)
            except TypeError:
                # Keep compatibility with a future MATE version that returns
                # a mutable dispatch id.
                return original_resolve_kernel_path(args, get_asm_id, mubin_dir)
            return _resolved_kernel_path(asm_id, str(Path(mubin_dir)))

        dispatcher_instance.resolve_kernel_path = cached_resolve_kernel_path
        dispatcher_instance._megatron_resolution_cache_installed = True
        return dispatcher_instance

    def cached_dispatcher(kernel_map_path):
        return _dispatcher_for_path(str(Path(kernel_map_path)))

    @functools.lru_cache(maxsize=None)
    def _verified_kernel(module: str, module_dir: str, kernel_file_name: str):
        return original_ensure_kernel(module, Path(module_dir), kernel_file_name)

    def cached_ensure_kernel(module, module_dir, kernel_file_name, repository=None):
        # Custom repositories may be mutable or non-hashable. They are not used
        # by the packaged mate-mubin path, so preserve original semantics.
        if repository is not None:
            return original_ensure_kernel(
                module, module_dir, kernel_file_name, repository=repository
            )
        return _verified_kernel(str(module), str(Path(module_dir)), str(kernel_file_name))

    gemm_api.MoeGemmMubinDispatcher = cached_dispatcher
    dispatch.ensure_mubin_kernel_artifact = cached_ensure_kernel
    gemm_api._megatron_mubin_dispatch_cache_installed = True
    return True


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
        print(f"[MATE_GROUPED_GEMM] {message}", flush=True)


class _MateGroupedLinear(torch.autograd.Function):
    """MATE ragged-M fprop/dgrad with Transformer Engine grouped wgrad."""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        counts: torch.Tensor,
        use_main_grad: bool,
        is_first_microbatch: bool | None,
        *weights: torch.Tensor,
    ):
        # In ``mate`` affinity mode this binds only the Python thread that
        # submits MATE work. DeepEP/communication threads were created before
        # this point and retain their unrestricted affinity.
        from .cpu_affinity import maybe_bind_local_rank_cpu_affinity

        maybe_bind_local_rank_cpu_affinity("mate")
        mate_gemm = load_mate_gemm()

        in_features = weights[0].shape[-1]
        out_features = weights[0].shape[0]
        flat_inp = inp.reshape(-1, in_features).contiguous()
        packed_weights = weights[0].as_strided(
            (len(weights), out_features, in_features),
            (out_features * in_features, in_features, 1),
        )
        out = torch.empty(
            (flat_inp.shape[0], out_features),
            dtype=flat_inp.dtype,
            device=flat_inp.device,
        )
        with torch.profiler.record_function("mate_grouped_gemm_fprop"):
            mate_gemm.ragged_m_moe_gemm_16bit(
                flat_inp,
                packed_weights,
                counts,
                out,
                gemm_mode="per_expert",
                major_a_mode="K",
                major_b_mode="K",
                backend="mubin",
            )

        ctx.inp_shape = inp.shape
        ctx.use_main_grad = use_main_grad
        ctx.is_first_microbatch = is_first_microbatch
        ctx.m_splits = tuple(counts._mate_m_splits)
        ctx.save_for_backward(flat_inp, counts, *weights)
        return out.reshape(*inp.shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mate_gemm = load_mate_gemm()

        flat_inp, counts, *weights = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        num_experts = len(weights)
        out_features, in_features = weights[0].shape
        packed_weights = weights[0].as_strided(
            (num_experts, out_features, in_features),
            (out_features * in_features, in_features, 1),
        )

        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = torch.empty_like(flat_inp)
            with torch.profiler.record_function("mate_grouped_gemm_dgrad"):
                mate_gemm.ragged_m_moe_gemm_16bit(
                    grad_output,
                    packed_weights,
                    counts,
                    dgrad,
                    gemm_mode="per_expert",
                    major_a_mode="K",
                    # Stored weights are [N, K]. dY[M, N] @ W[N, K]
                    # therefore consumes the same storage in N-major mode.
                    major_b_mode="N",
                    backend="mubin",
                )
            dgrad = dgrad.reshape(ctx.inp_shape)

        wgrads = [None] * num_experts
        if any(ctx.needs_input_grad[4:]):
            from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
            from transformer_engine.pytorch.module.base import (
                _2X_ACC_WGRAD,
                get_multi_stream_cublas_workspace,
            )

            if ctx.use_main_grad:
                wgrad_outputs = [weight.main_grad for weight in weights]
                grad_added_flags = [
                    getattr(weight, "grad_added_to_main_grad", None) for weight in weights
                ]
                if all(flag is not None for flag in grad_added_flags):
                    if len({bool(flag) for flag in grad_added_flags}) != 1:
                        raise RuntimeError(
                            "Grouped expert weights disagree on main_grad accumulation state"
                        )
                    # Megatron DDP resets this flag at the beginning of every
                    # optimizer iteration, so it is a reliable beta=0/beta=1
                    # signal even when non-FP8 modules do not reset TE's
                    # is_first_microbatch flag.
                    accumulate = bool(grad_added_flags[0])
                else:
                    accumulate = (
                        not ctx.is_first_microbatch
                        if ctx.is_first_microbatch is not None
                        else True
                    )
            else:
                wgrad_outputs = [torch.empty_like(weight) for weight in weights]
                accumulate = False

            input_mats = torch.split(flat_inp, ctx.m_splits)
            grad_output_mats = torch.split(grad_output, ctx.m_splits)
            with torch.profiler.record_function("mate_te_grouped_gemm_wgrad"):
                general_grouped_gemm(
                    list(input_mats),
                    list(grad_output_mats),
                    wgrad_outputs,
                    wgrad_outputs[0].dtype,
                    get_multi_stream_cublas_workspace(),
                    layout="NT",
                    m_splits=list(ctx.m_splits),
                    grad=True,
                    accumulate=accumulate,
                    use_split_accumulator=_2X_ACC_WGRAD,
                )

            if ctx.use_main_grad:
                # Megatron DDP must not add a second copy of this gradient.
                for weight in weights:
                    if hasattr(weight, "grad_added_to_main_grad"):
                        weight.grad_added_to_main_grad = True
            else:
                wgrads = wgrad_outputs

        return (dgrad, None, None, None, *wgrads)


def _is_packed(weights: Sequence[torch.Tensor]) -> bool:
    if not weights or not all(
        isinstance(weight, torch.Tensor) and weight.is_contiguous() for weight in weights
    ):
        return False
    bytes_per_weight = weights[0].numel() * weights[0].element_size()
    base = weights[0].data_ptr()
    return all(
        weight.shape == weights[0].shape
        and weight.dtype == weights[0].dtype
        and weight.device == weights[0].device
        and weight.data_ptr() == base + index * bytes_per_weight
        for index, weight in enumerate(weights)
    )


def _static_layout_supported(module, device: torch.device, use_main_grad: bool) -> bool:
    """Cache parameter/layout checks that are invariant during training.

    Only successful checks are cached. This lets a module created before DDP
    retry after ``main_grad`` and packed parameter storage have been installed.
    The cache stores booleans and device metadata, never parameters or grads.
    """
    cache_key = (bool(use_main_grad), device.type, device.index)
    cache = getattr(module, "_mate_static_support_cache", None)
    if cache is not None and cache.get(cache_key, False):
        return True

    weights = [getattr(module, f"weight{i}") for i in range(module.num_gemms)]
    supported = _is_packed(weights) and all(
        weight.dtype == torch.bfloat16 and weight.device == device for weight in weights
    )

    if supported and use_main_grad:
        main_grads = [getattr(weight, "main_grad", None) for weight in weights]
        supported = all(
            isinstance(grad, torch.Tensor)
            and grad.dtype == torch.float32
            and grad.device == device
            and grad.shape == weight.shape
            and grad.is_contiguous()
            for weight, grad in zip(weights, main_grads)
        )

    if supported:
        if cache is None:
            cache = {}
            module._mate_static_support_cache = cache
        cache[cache_key] = True
    return supported


def _pack_weights_before_ddp(module) -> None:
    """Pack expert Parameters before Megatron DDP remaps parameter storage."""
    weights = [getattr(module, f"weight{i}") for i in range(module.num_gemms)]
    if not weights or _is_packed(weights):
        return
    if any(
        weight.dtype != torch.bfloat16 or weight.device.type != "musa" for weight in weights
    ):
        return

    packed = torch.empty(
        (len(weights), *weights[0].shape),
        dtype=weights[0].dtype,
        device=weights[0].device,
    )
    with torch.no_grad():
        for index, weight in enumerate(weights):
            packed[index].copy_(weight)
            weight.data = packed[index]
    # Parameter views retain the allocation. Do not keep a second module-level
    # reference: distributed optimizer may later remap them into its own buffer.


def _supported(
    module,
    inp: torch.Tensor,
    counts: torch.Tensor,
    use_main_grad: bool,
) -> bool:
    if (
        module.fp8
        or module.fp8_calibration
        or module.apply_bias
        or module.return_bias
        or module.gemm_bias_unfused_add
    ):
        return False
    if module.save_original_input:
        return False
    if not use_main_grad and module.fuse_wgrad_accumulation:
        return False
    if not inp.is_contiguous() or inp.dtype != torch.bfloat16 or inp.device.type != "musa":
        return False
    if (
        not isinstance(counts, torch.Tensor)
        or counts.dtype != torch.int32
        or counts.device != inp.device
        or not counts.is_contiguous()
        or counts.numel() != module.num_gemms
        or not hasattr(counts, "_mate_m_splits")
        or len(counts._mate_m_splits) != module.num_gemms
        or sum(counts._mate_m_splits) != inp.reshape(-1, inp.shape[-1]).shape[0]
    ):
        return False

    return _static_layout_supported(module, inp.device, use_main_grad)


def _host_splits(m_splits):
    if not isinstance(m_splits, torch.Tensor):
        return m_splits
    if hasattr(m_splits, "_mate_m_splits"):
        return list(m_splits._mate_m_splits)
    return m_splits.tolist()


def install_mate_grouped_gemm() -> None:
    """Install the opt-in GroupedLinear patch before model construction."""
    import transformer_engine.pytorch as te

    grouped_linear = te.GroupedLinear
    if getattr(grouped_linear, "_mate_grouped_gemm_installed", False):
        return

    original_init = grouped_linear.__init__
    original_forward = grouped_linear.forward

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _pack_weights_before_ddp(self)

    @functools.wraps(original_forward)
    def patched_forward(
        self,
        inp,
        m_splits,
        is_first_microbatch=None,
        fine_grained_offload=False,
    ):
        use_main_grad = env_flag("MATE_USE_MAIN_GRAD", "1")
        if not _supported(self, inp, m_splits, use_main_grad) or fine_grained_offload:
            if not getattr(self, "_mate_fallback_logged", False):
                weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
                _log(
                    f"fallback num_gemms={self.num_gemms} dtype={inp.dtype} "
                    f"packed={_is_packed(weights)} use_main_grad={use_main_grad} "
                    f"counts_dtype={getattr(m_splits, 'dtype', None)} "
                    f"counts_device={getattr(m_splits, 'device', None)}"
                )
                self._mate_fallback_logged = True
            return original_forward(
                self,
                inp,
                _host_splits(m_splits),
                is_first_microbatch,
                fine_grained_offload,
            )

        if not getattr(self, "_mate_active_logged", False):
            shape = inp.reshape(-1, inp.shape[-1]).shape
            _log(
                f"active num_gemms={self.num_gemms} shape=[{shape[0]},{shape[1]}] "
                f"use_main_grad={use_main_grad}"
            )
            self._mate_active_logged = True

        with self.prepare_forward(inp, num_gemms=self.num_gemms) as prepared_inp:
            out = _MateGroupedLinear.apply(
                prepared_inp,
                m_splits,
                use_main_grad,
                is_first_microbatch,
                *[getattr(self, f"weight{i}") for i in range(self.num_gemms)],
            )
        if self.return_bias:
            return out, [None] * self.num_gemms
        return out

    grouped_linear.__init__ = patched_init
    grouped_linear.forward = patched_forward
    grouped_linear._mate_grouped_gemm_installed = True
    _log("installed: MATE fprop/dgrad + Transformer Engine grouped wgrad")
