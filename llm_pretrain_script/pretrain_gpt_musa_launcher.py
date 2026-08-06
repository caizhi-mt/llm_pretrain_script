#!/usr/bin/env python3
"""Launch Megatron-LM pretrain_gpt.py with musa_patch applied first."""

import os
import runpy
import sys

if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
    import musa_patch  # noqa: F401 — 必须最先导入，不可提前 import megatron
    import torch

    _orig_get_device_properties = torch.musa.get_device_properties

    def _patched_get_device_properties(device=None):
        if isinstance(device, str) and device.startswith("cuda"):
            device = device.replace("cuda", "musa")
        elif isinstance(device, torch.device) and device.type == "cuda":
            device = torch.device("musa", device.index)
        return _orig_get_device_properties(device)

    torch.musa.get_device_properties = _patched_get_device_properties
    torch.cuda.get_device_properties = _patched_get_device_properties

    import megatron.training.utils as megatron_training_utils
    import musa_patch.training as musa_training

    _noop_report_memory = lambda name: None
    megatron_training_utils.report_memory = _noop_report_memory
    musa_training.report_memory = _noop_report_memory

    def _get_device_arch_version_musa():
        return torch.cuda.get_device_properties(torch.device("musa:0")).major

    megatron_training_utils.get_device_arch_version = _get_device_arch_version_musa

    # core /home/Megatron-LM/pretrain_gpt.py 的 loss_func 返回
    # {'lm loss': Tensor([loss_sum, n_tokens])}；musa_patch NO_LOSS_REDUCE 只认 tuple/list，
    # 会把 2 元 Tensor 当标量累加，training_log 里 .item() 炸：
    # RuntimeError: a Tensor with 2 elements cannot be converted to Scalar
    _orig_training_log = musa_training.training_log

    def _training_log_core_compat(loss_dict, total_loss_dict, *args, **kwargs):
        for key, val in list(loss_dict.items()):
            if torch.is_tensor(val) and val.numel() == 2:
                loss_dict[key] = (val[0] / val[1].clamp(min=1)).detach()
            elif torch.is_tensor(val) and val.numel() > 1:
                loss_dict[key] = val.float().mean().detach()
        return _orig_training_log(loss_dict, total_loss_dict, *args, **kwargs)

    musa_training.training_log = _training_log_core_compat

# ---------------------------------------------------------------------------
# MoE router fusion: 绕过 Megatron 的 TE 版本门
#
# megatron/core/extensions/transformer_engine.py:1950 用
# is_te_min_version("2.7.0.dev") 决定要不要 import fused router 符号。摩尔线程
# 的 TE fork 已经实现了这套 API（transformer_engine/pytorch/router.py），但版本
# 号仍是 2.0.0，过不了这道门，三个符号被置成 None，运行到 router 前向时报
#     ValueError: fused_topk_with_score_function is not available.
#
# 不改 VERSION.txt 是因为那会连带打开 2.2~2.7 区间的另外约 22 处版本门，
# 引入一堆无关的未知数。这里只把这三个名字重新绑回去。
#
# ROUTER_FUSION_BYPASS=0 可关闭本段。
# ---------------------------------------------------------------------------
if os.getenv("ROUTER_FUSION_BYPASS", "1") == "1":
    try:
        from transformer_engine.pytorch.router import (
            fused_compute_score_for_moe_aux_loss as _te_fused_score,
            fused_moe_aux_loss as _te_fused_aux_loss,
            fused_topk_with_score_function as _te_fused_topk,
        )
    except ImportError as _e:
        print(f"[router-fusion] TE 无 pytorch.router，不做注入: {_e}", flush=True)
    else:
        import megatron.core.extensions.transformer_engine as _te_ext
        from megatron.core.transformer.moe import moe_utils as _mu

        _patched = []
        for _mod in (_te_ext, _mu):
            for _name, _fn in (
                ("fused_topk_with_score_function", _te_fused_topk),
                ("fused_compute_score_for_moe_aux_loss", _te_fused_score),
                ("fused_moe_aux_loss", _te_fused_aux_loss),
            ):
                if getattr(_mod, _name, None) is None:
                    setattr(_mod, _name, _fn)
                    _patched.append(f"{_mod.__name__}.{_name}")
        # moe_utils 的 fused 分支还会先看 HAVE_TE
        if not getattr(_mu, "HAVE_TE", False):
            _mu.HAVE_TE = True
            _patched.append("moe_utils.HAVE_TE")
        print(f"[router-fusion] 注入 {len(_patched)} 个符号: {_patched}", flush=True)


mcore_path = os.environ.get("MCORE_PATH", "/home/Megatron-LM")
pretrain_script = os.environ.get(
    "PRETRAIN_SCRIPT",
    os.path.join(mcore_path, "pretrain_gpt.py"),
)

if not os.path.isfile(pretrain_script):
    raise FileNotFoundError(f"pretrain_gpt.py not found: {pretrain_script}")

sys.argv[0] = pretrain_script
runpy.run_path(pretrain_script, run_name="__main__")
