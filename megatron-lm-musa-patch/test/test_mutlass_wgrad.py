"""CPU-only policy tests for the optional MP31 MUTLASS wgrad path."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "musa_patch" / "mutlass_wgrad.py"
SPEC = importlib.util.spec_from_file_location("mutlass_wgrad_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def select(**overrides):
    arguments = dict(
        num_experts=128,
        total_tokens=8192,
        out_features=1536,
        in_features=2048,
        max_tokens=101,
        use_main_grad=True,
        accumulate=True,
    )
    arguments.update(overrides)
    return MODULE.select_mutlass_wgrad(**arguments)


def test_selects_benchmarked_nonuniform_small_m_shapes():
    assert select() == "gate_up"
    assert select(out_features=2048, in_features=768) == "down"


def test_selects_actual_e32_large_m_fc1_and_fc2():
    assert (
        select(
            num_experts=32,
            total_tokens=65536,
            out_features=4096,
            in_features=7168,
            max_tokens=3060,
        )
        == "e32_gate_up"
    )
    assert (
        select(
            num_experts=32,
            total_tokens=65536,
            out_features=7168,
            in_features=2048,
            max_tokens=3060,
        )
        == "e32_down"
    )


def test_first_microbatch_and_non_main_grad_stay_on_te():
    assert select(accumulate=False) is None
    assert select(use_main_grad=False) is None


def test_unbenchmarked_expert_and_feature_shapes_stay_on_te():
    assert select(num_experts=16) is None
    assert select(out_features=4096, in_features=7168) is None


def test_large_average_or_long_tail_stays_on_te():
    assert select(total_tokens=128 * 513) is None
    assert select(max_tokens=1025) is None


def test_alignment_helper_rejects_contiguous_offset_view():
    storage = torch.empty(17, dtype=torch.bfloat16)
    offset = storage[1:]
    assert offset.is_contiguous()
    assert not MODULE._aligned(offset, 4)
    assert MODULE._aligned(torch.empty(16, dtype=torch.float32), 8)


@pytest.mark.parametrize(
    "n_features,k_features",
    [(256, 2048), (256, 64), (4096, 7168), (7168, 2048)],
)
def test_native_two_microbatch_accumulation_on_mp31(n_features, k_features):
    if os.getenv("RUN_MUTLASS_WGRAD_MUSA_TESTS", "0") != "1":
        pytest.skip("set RUN_MUTLASS_WGRAD_MUSA_TESTS=1 for the MP31 test")

    import torch_musa  # noqa: F401
    from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
    from transformer_engine.pytorch.module.base import (
        _2X_ACC_WGRAD,
        get_multi_stream_cublas_workspace,
    )

    if not MODULE.is_mp31_device():
        pytest.skip("native wgrad requires MP31")

    device = torch.device("musa:0")
    torch.musa.set_device(device)
    split_sets = ([5, 6, 8], [4, 0, 7])
    microbatches = []
    for index, splits in enumerate(split_sets):
        torch.manual_seed(1200 + index)
        total = sum(splits)
        x = torch.randn(total, k_features, device=device, dtype=torch.bfloat16)
        dy = torch.randn(total, n_features, device=device, dtype=torch.bfloat16)
        microbatches.append((splits, x, dy))

    reference = [
        torch.empty(n_features, k_features, device=device, dtype=torch.float32)
        for _ in split_sets[0]
    ]
    actual = [torch.empty_like(output) for output in reference]
    workspace = get_multi_stream_cublas_workspace()

    def te(index, outputs, accumulate):
        splits, x, dy = microbatches[index]
        general_grouped_gemm(
            list(torch.split(x, splits)),
            list(torch.split(dy, splits)),
            outputs,
            torch.float32,
            workspace,
            layout="NT",
            m_splits=list(splits),
            grad=True,
            accumulate=accumulate,
            use_split_accumulator=_2X_ACC_WGRAD,
        )

    te(0, reference, False)
    te(1, reference, True)

    nondefault = torch.musa.Stream()
    nondefault.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(nondefault):
        te(0, actual, False)
        splits, x, dy = microbatches[1]
        MODULE.mutlass_wgrad_accumulate(x, dy, splits, actual)
    torch.musa.current_stream().wait_stream(nondefault)
    torch.musa.synchronize()

    for result, expected in zip(actual, reference):
        torch.testing.assert_close(result, expected, atol=2.0e-5, rtol=2.0e-5)


@pytest.mark.parametrize("n_features,k_features", [(4096, 7168), (7168, 2048)])
def test_e32_k_grouped_cross_launch_and_zero_expert_on_mp31(n_features, k_features):
    if os.getenv("RUN_MUTLASS_WGRAD_MUSA_TESTS", "0") != "1":
        pytest.skip("set RUN_MUTLASS_WGRAD_MUSA_TESTS=1 for the MP31 test")

    import torch_musa  # noqa: F401
    from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
    from transformer_engine.pytorch.module.base import (
        _2X_ACC_WGRAD,
        get_multi_stream_cublas_workspace,
    )

    if not MODULE.is_mp31_device():
        pytest.skip("native wgrad requires MP31")

    device = torch.device("musa:0")
    torch.musa.set_device(device)
    # Five active experts force a second group4 launch; the zero expert also
    # verifies that beta=1 leaves its existing main_grad untouched.
    splits = [61, 0, 65, 64, 63, 67]
    torch.manual_seed(1701)
    x = torch.randn(sum(splits), k_features, device=device, dtype=torch.bfloat16)
    dy = torch.randn(sum(splits), n_features, device=device, dtype=torch.bfloat16)
    reference = [
        torch.randn(n_features, k_features, device=device, dtype=torch.float32)
        for _ in splits
    ]
    actual = [output.clone() for output in reference]
    zero_initial = actual[1].clone()

    general_grouped_gemm(
        list(torch.split(x, splits)),
        list(torch.split(dy, splits)),
        reference,
        torch.float32,
        get_multi_stream_cublas_workspace(),
        layout="NT",
        m_splits=splits,
        grad=True,
        accumulate=True,
        use_split_accumulator=_2X_ACC_WGRAD,
    )
    MODULE.mutlass_wgrad_accumulate(x, dy, splits, actual)
    torch.musa.synchronize()

    for result, expected in zip(actual, reference):
        torch.testing.assert_close(result, expected, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(actual[1], zero_initial, atol=0, rtol=0)
