# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
from collections import deque

import torch
from deep_ep.utils import EventHandle, EventOverlap

from megatron.core.transformer.moe.fused_a2a import (
    get_buffer,
    get_hidden_bytes,
)
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from megatron.core.tensor_parallel.random import get_checkpoint_phase

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import (
        fused_permute_with_probs,
        fused_unpermute,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


_DEEPEP_CACHE_LOGGED = False
_POST_FC1_CAPTURE_ATTR = "_deepep_recompute_cache_capture_after_fc1"


def _env_flag(name: str, default: str) -> bool:
    value = os.getenv(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _capture_after_fc1_enabled(manager) -> bool:
    if not _env_flag("DEEPEP_CACHE_CAPTURE_AFTER_FC1", "1"):
        return False
    config = manager.config
    return (
        config.moe_grouped_gemm
        and not getattr(config, "moe_use_legacy_grouped_gemm", False)
        and config.transformer_impl == "transformer_engine"
    )


def _recompute_cache_configured(manager) -> bool:
    if not _env_flag("DEEPEP_CACHE_RECOMPUTE_DISPATCH", "1"):
        return False
    # This bridge targets the production BF16 path. TE owns FP8 checkpointing.
    if getattr(manager.config, "fp8", None):
        return False
    granularity = getattr(manager.config, "recompute_granularity", None)
    recompute_modules = getattr(manager.config, "recompute_modules", None) or []
    return granularity == "full" or (
        granularity == "selective" and "moe" in recompute_modules
    )


def _clone_cache_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_cache_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_cache_value(item) for item in value]
    return value


def _retain_dispatch_handle(handle):
    """Snapshot the ACE layout before the original forward consumes it."""
    if not isinstance(handle, tuple) or len(handle) != 5:
        raise RuntimeError("unexpected DeepEP ACE dispatch handle")
    global_stride_list, row_id_map, num_recv_tokens, num_tokens, num_experts = handle
    # The original forward still consumes its handle in combine. Snapshot the
    # row map so recompute owns an independent layout.
    return (
        list(global_stride_list),
        row_id_map.detach().clone(),
        num_recv_tokens,
        num_tokens,
        num_experts,
    )


def _cache_tensor_bytes(value, seen=None):
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_cache_tensor_bytes(item, seen) for item in value)
    if isinstance(value, dict):
        return sum(_cache_tensor_bytes(item, seen) for item in value.values())
    return 0


def _get_recompute_cache(manager):
    cache = getattr(manager, "_deepep_recompute_dispatch_cache", None)
    if cache is None:
        cache = {}
        manager._deepep_recompute_dispatch_cache = cache
    return cache


def _get_recompute_cache_stream(manager, device):
    stream = getattr(manager, "_deepep_recompute_cache_stream", None)
    if stream is None:
        stream = torch.musa.Stream(device=device)
        manager._deepep_recompute_cache_stream = stream
    return stream


def _cache_key_matches(manager, entry, hidden_states, async_finish, allocate_on_comm_stream):
    return (
        entry.get("finalized", False)
        and entry["hidden_shape"] == tuple(hidden_states.shape)
        and entry["hidden_dtype"] == hidden_states.dtype
        and entry["hidden_device"] == hidden_states.device
        and entry["token_indices_shape"] == tuple(manager.token_indices.shape)
        and entry["token_probs_shape"] == tuple(manager.token_probs.shape)
        and entry["token_probs_dtype"] == manager.token_probs.dtype
        and entry["async_finish"] == async_finish
        and entry["allocate_on_comm_stream"] == allocate_on_comm_stream
    )


def _global_validation_mismatch(manager, entry) -> bool:
    local_mismatch = not torch.equal(entry["token_indices"], manager.token_indices)
    local_mismatch = local_mismatch or not torch.equal(
        entry["token_probs"], manager.token_probs.detach()
    )
    if not torch.distributed.is_initialized():
        return local_mismatch
    mismatch = torch.tensor(
        int(local_mismatch), dtype=torch.int32, device=manager.token_indices.device
    )
    torch.distributed.all_reduce(mismatch, op=torch.distributed.ReduceOp.MAX, group=manager.group)
    return bool(mismatch.item())


def _capture_recompute_dispatch(
    manager,
    checkpoint_id,
    hidden_states,
    async_finish,
    allocate_on_comm_stream,
):
    cache = _get_recompute_cache(manager)
    max_entries = int(os.getenv("DEEPEP_CACHE_RECOMPUTE_MAX_ENTRIES", "256"))
    if max_entries <= 0:
        raise ValueError("DEEPEP_CACHE_RECOMPUTE_MAX_ENTRIES must be positive")
    entry_count = sum(len(entries) for entries in cache.values())
    if entry_count >= max_entries:
        raise RuntimeError(
            "DeepEP recompute dispatch cache exceeded "
            f"DEEPEP_CACHE_RECOMPUTE_MAX_ENTRIES={max_entries}"
        )
    if getattr(manager, "_deepep_recompute_capture_record", None) is not None:
        raise RuntimeError("DeepEP recompute cache capture was not finalized")

    validate = _env_flag("DEEPEP_CACHE_RECOMPUTE_VALIDATE", "0")
    if _capture_after_fc1_enabled(manager):
        entry = {
            "source_handle": manager.handle,
            "source_dispatched_indices": manager.dispatched_indices,
            "source_dispatched_probs": manager.dispatched_probs,
            "source_num_tokens_per_rank": manager.num_tokens_per_rank,
            "source_tokens_per_expert": manager.tokens_per_expert,
            "hidden_shape": tuple(hidden_states.shape),
            "hidden_dtype": hidden_states.dtype,
            "hidden_device": hidden_states.device,
            "token_indices_shape": tuple(manager.token_indices.shape),
            "token_probs_shape": tuple(manager.token_probs.shape),
            "token_probs_dtype": manager.token_probs.dtype,
            "async_finish": async_finish,
            "allocate_on_comm_stream": allocate_on_comm_stream,
            "checkpoint_id": checkpoint_id,
            "validate": validate,
            "finalized": False,
            "capture_after_fc1": True,
        }
        if validate:
            entry["source_token_indices"] = manager.token_indices
            entry["source_token_probs"] = manager.token_probs
        manager._deepep_recompute_capture_record = entry
        manager._deepep_recompute_cache_captures = (
            getattr(manager, "_deepep_recompute_cache_captures", 0) + 1
        )
        return

    current_stream = torch.musa.current_stream(hidden_states.device)
    cache_stream = _get_recompute_cache_stream(manager, hidden_states.device)
    source_ready = torch.musa.Event()
    source_ready.record(current_stream)
    cache_stream.wait_event(source_ready)
    with torch.musa.stream(cache_stream):
        entry = {
            "handle": _retain_dispatch_handle(manager.handle),
            "dispatched_indices": _clone_cache_value(manager.dispatched_indices),
            "dispatched_probs": _clone_cache_value(manager.dispatched_probs),
            # These tensors are immutable, owning outputs. Keeping references
            # avoids launching tiny D2D copies whose setup dominates payload.
            "tokens_per_expert": manager.tokens_per_expert,
            "num_tokens_per_rank": manager.num_tokens_per_rank,
            "hidden_shape": tuple(hidden_states.shape),
            "hidden_dtype": hidden_states.dtype,
            "hidden_device": hidden_states.device,
            "token_indices_shape": tuple(manager.token_indices.shape),
            "token_probs_shape": tuple(manager.token_probs.shape),
            "token_probs_dtype": manager.token_probs.dtype,
            "async_finish": async_finish,
            "allocate_on_comm_stream": allocate_on_comm_stream,
            "checkpoint_id": checkpoint_id,
            "finalized": False,
        }
        if validate:
            entry["token_indices"] = manager.token_indices.detach().clone()
            entry["token_probs"] = manager.token_probs.detach().clone()
        entry["ready_event"] = torch.musa.Event()
        entry["ready_event"].record(cache_stream)
    cache.setdefault(checkpoint_id, deque()).append(entry)
    manager._deepep_recompute_capture_record = entry
    manager._deepep_recompute_cache_captures = (
        getattr(manager, "_deepep_recompute_cache_captures", 0) + 1
    )


def _complete_recompute_dispatch_capture_after_fc1(manager, entry):
    """Clone the routing cache after FC1 has been submitted to the compute stream."""
    global _DEEPEP_CACHE_LOGGED

    if getattr(manager, "_deepep_recompute_capture_record", None) is not entry:
        raise RuntimeError("DeepEP post-FC1 cache capture record changed before completion")
    if entry.get("finalized", False):
        raise RuntimeError("DeepEP post-FC1 cache capture completed more than once")

    with torch.profiler.record_function("deepep_recompute_cache_capture_after_fc1"):
        retained_handle = _retain_dispatch_handle(entry.pop("source_handle"))
        cached_indices = entry.pop("source_dispatched_indices").detach().clone()
        cached_probs = entry.pop("source_dispatched_probs").detach().clone()
        # These are immutable, owning outputs, matching the existing capture
        # path. Keep references rather than launching tiny D2D copies.
        cached_tokens_per_expert = entry.pop("source_tokens_per_expert")
        cached_num_tokens_per_rank = entry.pop("source_num_tokens_per_rank")
        if entry.pop("validate"):
            entry["token_indices"] = entry.pop("source_token_indices").detach().clone()
            entry["token_probs"] = entry.pop("source_token_probs").detach().clone()
        ready_event = torch.musa.Event()
        ready_event.record(torch.musa.current_stream())

    entry.update(
        {
            "handle": retained_handle,
            "dispatched_indices": cached_indices,
            "dispatched_probs": cached_probs,
            "tokens_per_expert": cached_tokens_per_expert,
            "num_tokens_per_rank": cached_num_tokens_per_rank,
            "ready_event": ready_event,
            "capture_stream": None,
            "finalized": True,
        }
    )
    entry.pop("capture_after_fc1", None)
    checkpoint_id = entry.pop("checkpoint_id")
    cache = _get_recompute_cache(manager)
    cache.setdefault(checkpoint_id, deque()).append(entry)
    manager._deepep_recompute_capture_record = None

    if not _DEEPEP_CACHE_LOGGED:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(
                "[DEEPEP_RECOMPUTE_CACHE] capture enabled "
                f"validate={entry.get('token_indices') is not None} "
                f"capture_after_fc1=True bytes_per_entry={_cache_tensor_bytes(entry)}",
                flush=True,
            )
        _DEEPEP_CACHE_LOGGED = True


def _finalize_recompute_dispatch_capture(manager):
    global _DEEPEP_CACHE_LOGGED
    entry = getattr(manager, "_deepep_recompute_capture_record", None)
    if entry is None:
        return
    if entry.get("capture_after_fc1", False):
        counts = manager.tokens_per_expert
        if hasattr(counts, _POST_FC1_CAPTURE_ATTR):
            raise RuntimeError("DeepEP post-FC1 cache capture callback already exists")
        entry["source_tokens_per_expert"] = counts

        def capture_after_fc1():
            _complete_recompute_dispatch_capture_after_fc1(manager, entry)

        setattr(counts, _POST_FC1_CAPTURE_ATTR, capture_after_fc1)
        return
    entry["tokens_per_expert"] = manager.tokens_per_expert
    entry["finalized"] = True
    manager._deepep_recompute_capture_record = None
    # The snapshot normally finishes under the permutation kernels. Preserve
    # correctness for short expert paths without introducing a host sync.
    entry["ready_event"].wait(torch.musa.current_stream())

    if not _DEEPEP_CACHE_LOGGED and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        _DEEPEP_CACHE_LOGGED = True
        print(
            "[DEEPEP_RECOMPUTE_CACHE] enabled "
            f"bytes_per_entry={_cache_tensor_bytes(entry)}",
            flush=True,
        )


class _CachedRecomputeDispatch(torch.autograd.Function):
    """Redispatch recomputed hidden states with a cached DeepEP layout handle."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        token_probs,
        entry,
        group,
        async_finish,
        allocate_on_comm_stream,
    ):
        handle = entry["handle"]
        previous_event = EventOverlap(EventHandle()) if async_finish else None
        buffer = get_buffer(group, get_hidden_bytes(hidden_states))
        (
            dispatched_hidden_states,
            recv_indices,
            recv_probs,
            recv_tokens_per_expert,
            recv_handle,
            after_event,
        ) = buffer.dispatch(
            hidden_states.contiguous(),
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()
        if any(
            value is not None
            for value in (recv_indices, recv_probs, recv_tokens_per_expert, recv_handle)
        ):
            raise RuntimeError("cached DeepEP dispatch unexpectedly returned routing metadata")

        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        dispatched_indices = entry["dispatched_indices"]
        dispatched_probs = entry["dispatched_probs"]
        tokens_per_expert = entry["tokens_per_expert"]
        num_tokens_per_rank = entry["num_tokens_per_rank"]
        non_differentiable = [dispatched_indices, tokens_per_expert]
        if isinstance(num_tokens_per_rank, torch.Tensor):
            non_differentiable.append(num_tokens_per_rank)
        ctx.mark_non_differentiable(*non_differentiable)
        return (
            dispatched_hidden_states,
            dispatched_indices,
            dispatched_probs,
            tokens_per_expert,
            num_tokens_per_rank,
            handle,
        )

    @staticmethod
    def backward(
        ctx,
        grad_hidden_states,
        grad_dispatched_indices,
        grad_dispatched_probs,
        grad_tokens_per_expert,
        grad_num_tokens_per_rank,
        grad_handle,
    ):
        del (
            grad_dispatched_indices,
            grad_tokens_per_expert,
            grad_num_tokens_per_rank,
            grad_handle,
        )
        previous_event = EventOverlap(EventHandle()) if ctx.async_finish else None
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_hidden_states))
        topk_weight_grads = (
            grad_dispatched_probs.float() if grad_dispatched_probs is not None else None
        )
        grad_hidden_states, grad_token_probs, after_event = buffer.combine(
            grad_hidden_states.contiguous(),
            ctx.handle,
            topk_weights=topk_weight_grads,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return (
            grad_hidden_states,
            grad_token_probs,
            None,
            None,
            None,
            None,
        )


def _DeepepManager_get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor):
    deepep_buffer = get_buffer(self.group, get_hidden_bytes(hidden_states))
    ace_hidden_states, _ = deepep_buffer.get_ace_combine_buffer(
        self.hidden_shape_before_permute[0],
        self.hidden_shape_before_permute[1],
        1,
        False,
    )

    if not HAVE_TE or fused_unpermute is None:
        raise ValueError("fused_unpermute is not available. Please install TE >= 2.1.0.")
    hidden_states = fused_unpermute(
        hidden_states,
        self.reversed_mapping_for_combine,
        restore_shape=self.hidden_shape_before_permute,
        preallocated_act_f=ace_hidden_states,
    )

    # Custom expert specs may bypass TEGroupedMLP.forward even when the
    # configuration advertises grouped GEMM. Complete the pending snapshot
    # here before DeepEP combine consumes the original handle.
    pending_capture = getattr(self, "_deepep_recompute_capture_record", None)
    if pending_capture is not None and pending_capture.get("capture_after_fc1", False):
        counts = self.tokens_per_expert
        capture_after_fc1 = getattr(counts, _POST_FC1_CAPTURE_ATTR, None)
        if capture_after_fc1 is not None:
            try:
                capture_after_fc1()
            finally:
                delattr(counts, _POST_FC1_CAPTURE_ATTR)
        else:
            _complete_recompute_dispatch_capture_after_fc1(self, pending_capture)

    return hidden_states


def _DeepepManager_get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor):
    deepep_buffer =  get_buffer(self.group, get_hidden_bytes(hidden_states))

    ace_hidden_states, ace_probs = deepep_buffer.get_ace_combine_buffer(
        hidden_states.size(0), hidden_states.size(1), self.router_topk, True)

    host_splits = getattr(self.tokens_per_expert, "_mate_m_splits", None)
    self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(
        self.dispatched_indices,
        self.dispatched_probs,
        self.num_local_experts,
        preallocated_probs_b=ace_probs,
    )

    if getattr(self.tokens_per_expert, "_mate_deferred_device_counts", False):
        # The routing map already contains the exact per-expert membership on
        # device.  A column reduction is much cheaper than contended atomics
        # and avoids both the pageable H2D copy and a CPU-visible item() sync.
        # MUSA's large-column bool reduction is inaccurate for the current
        # dispatched-token shape (~32K x 32).  Keep each reduction below the
        # backend threshold, then reduce the tiny partial-count matrix.
        count_chunks = self.dispatched_routing_map.split(8192, dim=0)
        partial_counts = [chunk.sum(dim=0, dtype=torch.int32) for chunk in count_chunks]
        device_counts = torch.stack(partial_counts, dim=0).sum(dim=0, dtype=torch.int32)
        device_counts._mate_m_splits = host_splits
        self.tokens_per_expert = device_counts

    # if self.config.moe_router_padding_for_fp8:
    #     self.dispatched_routing_map, self.tokens_per_expert = self._pad_routing_map(
    #         self.dispatched_routing_map, self.tokens_per_expert
    #     )

    self.hidden_shape_before_permute = hidden_states.shape
    assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"

    hidden_states, permuted_probs, self.reversed_mapping_for_combine = fused_permute_with_probs(
        hidden_states,
        self.dispatched_probs,
        self.dispatched_routing_map,
        num_out_tokens=(
            sum(host_splits) if host_splits is not None else self.tokens_per_expert.sum().item()
        ),
        preallocated_act_b=ace_hidden_states,
    )

    if self.router_dtype == "fp64":
        permuted_probs = permuted_probs.to(torch.float64)
    _finalize_recompute_dispatch_capture(self)
    return hidden_states, permuted_probs


from transformer_engine.musa.pytorch.utils import replace_attr
from megatron.core.transformer.moe.token_dispatcher import _DeepepManager


_ORIGINAL_DEEPEP_DISPATCH = _DeepepManager.dispatch


def _DeepepManager_dispatch_with_recompute_cache(
    self,
    hidden_states: torch.Tensor,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
):
    if not _recompute_cache_configured(self):
        return _ORIGINAL_DEEPEP_DISPATCH(
            self, hidden_states, async_finish, allocate_on_comm_stream
        )

    checkpoint_context = get_checkpoint_phase()
    if checkpoint_context is None:
        return _ORIGINAL_DEEPEP_DISPATCH(
            self, hidden_states, async_finish, allocate_on_comm_stream
        )
    checkpoint_phase, checkpoint_id = checkpoint_context
    if checkpoint_phase == "forward":
        output = _ORIGINAL_DEEPEP_DISPATCH(
            self, hidden_states, async_finish, allocate_on_comm_stream
        )
        _capture_recompute_dispatch(
            self,
            checkpoint_id,
            hidden_states,
            async_finish,
            allocate_on_comm_stream,
        )
        return output

    if checkpoint_phase != "recompute":
        return _ORIGINAL_DEEPEP_DISPATCH(
            self, hidden_states, async_finish, allocate_on_comm_stream
        )

    cache = _get_recompute_cache(self)
    checkpoint_entries = cache.get(checkpoint_id)
    if not checkpoint_entries:
        self._deepep_recompute_cache_misses = (
            getattr(self, "_deepep_recompute_cache_misses", 0) + 1
        )
        return _ORIGINAL_DEEPEP_DISPATCH(
            self, hidden_states, async_finish, allocate_on_comm_stream
        )

    entry = checkpoint_entries.popleft()
    if not checkpoint_entries:
        del cache[checkpoint_id]
    entry["ready_event"].wait(torch.musa.current_stream(hidden_states.device))
    key_matches = _cache_key_matches(
        self, entry, hidden_states, async_finish, allocate_on_comm_stream
    )
    validation_mismatch = False
    if key_matches and _env_flag("DEEPEP_CACHE_RECOMPUTE_VALIDATE", "0"):
        validation_mismatch = _global_validation_mismatch(self, entry)

    if not key_matches:
        raise RuntimeError(
            "DeepEP recompute cache key mismatch across a checkpoint replay; "
            "refusing to mix full and handle dispatch protocols"
        )

    if validation_mismatch:
        self._deepep_recompute_cache_fallbacks = (
            getattr(self, "_deepep_recompute_cache_fallbacks", 0) + 1
        )
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(
                "[DEEPEP_RECOMPUTE_CACHE] fallback "
                "route mismatch; using full dispatch",
                flush=True,
            )
        return _ORIGINAL_DEEPEP_DISPATCH(
            self, hidden_states, async_finish, allocate_on_comm_stream
        )

    with torch.profiler.record_function("deepep_recompute_cache_hit"):
        (
            dispatched_hidden_states,
            self.dispatched_indices,
            self.dispatched_probs,
            self.tokens_per_expert,
            self.num_tokens_per_rank,
            self.handle,
        ) = _CachedRecomputeDispatch.apply(
            hidden_states,
            self.token_probs.float(),
            entry,
            self.group,
            async_finish,
            allocate_on_comm_stream,
        )

    self._deepep_recompute_cache_hits = getattr(self, "_deepep_recompute_cache_hits", 0) + 1
    if self._deepep_recompute_cache_hits == 1 and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        print("[DEEPEP_RECOMPUTE_CACHE] first cache hit", flush=True)
    return dispatched_hidden_states


replace_attr(
    _DeepepManager,
    "dispatch",
    _DeepepManager_dispatch_with_recompute_cache,
)
replace_attr(
    _DeepepManager,
    "get_restored_hidden_states_by_experts",
    _DeepepManager_get_restored_hidden_states_by_experts,
)
replace_attr(
    _DeepepManager,
    "get_permuted_hidden_states_by_experts",
    _DeepepManager_get_permuted_hidden_states_by_experts,
)
