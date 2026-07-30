import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "Megatron-LM"))
from megatron.core.transformer import multi_latent_attention as mla


MODULE_PATH = Path(__file__).parents[1] / "musa_patch" / "rotary_pos_embedding.py"
SPEC = importlib.util.spec_from_file_location("native_rope_test_module", MODULE_PATH)
rope = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rope)


def _config(**overrides):
    values = {
        "apply_rope_fusion": False,
        "multi_latent_attention": True,
        "rope_type": "rope",
        "rotary_interleaved": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _mla_config(**overrides):
    values = {
        "context_parallel_size": 1,
        "qk_head_dim": 128,
        "qk_pos_emb_head_dim": 64,
        "rope_type": "rope",
        "rotary_interleaved": False,
        "v_head_dim": 128,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_native_rope_is_used_from_unfused_standard_rope_path(monkeypatch):
    t = torch.randn(4, 1, 2, 8)
    freqs = torch.randn(4, 1, 1, 8)
    sentinel = object()
    call = {}

    monkeypatch.setattr(rope, "_can_use_native_rope", lambda *args: True)

    def fake_rope(input_, freqs_, **kwargs):
        call["input"] = input_
        call["freqs"] = freqs_
        call.update(kwargs)
        return sentinel

    monkeypatch.setattr(torch, "rope", fake_rope)
    result = rope.apply_rotary_pos_emb(t, freqs, _config(), cp_group=object())

    assert result is sentinel
    assert call["input"] is t
    assert call["freqs"].shape == (4, 8)
    assert call["rotary_interleaved"] is False
    assert call["batch_first"] is False
    assert call["multi_latent_attention"] is True


def test_native_rope_default_and_explicit_fallback(monkeypatch):
    t = torch.randn(4, 1, 2, 8)
    freqs = torch.randn(4, 1, 1, 8)

    monkeypatch.delenv("MUSA_NATIVE_ROPE", raising=False)
    monkeypatch.setattr(rope, "_is_musa_tensor", lambda _: True)
    assert rope._can_use_native_rope(t, freqs, _config(), None, 1.0)

    monkeypatch.setenv("MUSA_NATIVE_ROPE", "0")
    assert not rope._can_use_native_rope(t, freqs, _config(), None, 1.0)


def test_native_rope_rejects_unsupported_call_shapes(monkeypatch):
    t = torch.randn(4, 1, 2, 8)
    freqs = torch.randn(4, 1, 1, 8)
    monkeypatch.setenv("MUSA_NATIVE_ROPE", "1")
    monkeypatch.setattr(rope, "_is_musa_tensor", lambda _: True)

    assert not rope._can_use_native_rope(t, freqs, _config(rope_type="yarn"), None, 1.0)
    assert not rope._can_use_native_rope(t, freqs, _config(context_parallel_size=2), None, 1.0)
    assert not rope._can_use_native_rope(t, freqs, _config(), torch.tensor([0, 4]), 1.0)
    assert not rope._can_use_native_rope(t, freqs, _config(), None, 0.5)
    assert not rope._can_use_native_rope(t, freqs.squeeze(1), _config(), None, 1.0)

    monkeypatch.setattr(rope, "_is_musa_tensor", lambda tensor: tensor is t)
    assert not rope._can_use_native_rope(t, freqs, _config(), None, 1.0)


def test_invalid_native_rope_flag_fails_fast(monkeypatch):
    monkeypatch.setenv("MUSA_NATIVE_ROPE", "sometimes")
    with pytest.raises(ValueError, match="MUSA_NATIVE_ROPE must be a boolean flag"):
        rope._env_flag("MUSA_NATIVE_ROPE", "1")


def test_musa_fused_mla_rope_default_and_fallbacks(monkeypatch):
    hidden_states = SimpleNamespace(dtype=torch.bfloat16, is_musa=True)
    monkeypatch.delenv("MUSA_NATIVE_ROPE", raising=False)
    monkeypatch.delenv("MUSA_FUSED_MLA_ROPE", raising=False)
    monkeypatch.setattr(mla, "fused_apply_mla_rope_for_q", object())
    monkeypatch.setattr(mla, "fused_apply_mla_rope_for_kv", object())

    assert mla._can_use_musa_fused_mla_rope(
        _mla_config(), hidden_states, packed_seq=False, inference_context=None
    )

    monkeypatch.setenv("MUSA_FUSED_MLA_ROPE", "0")
    assert not mla._can_use_musa_fused_mla_rope(
        _mla_config(), hidden_states, packed_seq=False, inference_context=None
    )
    monkeypatch.setenv("MUSA_FUSED_MLA_ROPE", "1")
    monkeypatch.setenv("MUSA_NATIVE_ROPE", "0")
    assert not mla._can_use_musa_fused_mla_rope(
        _mla_config(), hidden_states, packed_seq=False, inference_context=None
    )


def test_invalid_musa_fused_mla_rope_flag_fails_fast(monkeypatch):
    hidden_states = SimpleNamespace(dtype=torch.bfloat16, is_musa=True)
    monkeypatch.setenv("MUSA_NATIVE_ROPE", "1")
    monkeypatch.setenv("MUSA_FUSED_MLA_ROPE", "sometimes")

    with pytest.raises(ValueError, match="MUSA_FUSED_MLA_ROPE must be a boolean flag"):
        mla._can_use_musa_fused_mla_rope(
            _mla_config(), hidden_states, packed_seq=False, inference_context=None
        )


@pytest.mark.parametrize(
    ("config_overrides", "packed_seq", "inference_context"),
    [
        ({"rope_type": "yarn"}, False, None),
        ({"context_parallel_size": 2}, False, None),
        ({"rotary_interleaved": True}, False, None),
        ({"qk_head_dim": 96}, False, None),
        ({}, True, None),
        ({}, False, object()),
    ],
)
def test_musa_fused_mla_rope_rejects_unsupported_modes(
    monkeypatch, config_overrides, packed_seq, inference_context
):
    hidden_states = SimpleNamespace(dtype=torch.bfloat16, is_musa=True)
    monkeypatch.setenv("MUSA_NATIVE_ROPE", "1")
    monkeypatch.setenv("MUSA_FUSED_MLA_ROPE", "1")
    monkeypatch.setattr(mla, "fused_apply_mla_rope_for_q", object())
    monkeypatch.setattr(mla, "fused_apply_mla_rope_for_kv", object())

    assert not mla._can_use_musa_fused_mla_rope(
        _mla_config(**config_overrides), hidden_states, packed_seq, inference_context
    )
