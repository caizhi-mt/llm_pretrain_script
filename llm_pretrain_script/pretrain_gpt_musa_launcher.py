#!/usr/bin/env python3
"""Launch Megatron-LM pretrain_gpt.py with musa_patch applied first."""

import os
import runpy
import sys
from pathlib import Path

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

repo_root = Path(__file__).resolve().parent.parent
mcore_path = os.environ.get("MCORE_PATH", str(repo_root / "Megatron-LM"))
pretrain_script = os.environ.get(
    "PRETRAIN_SCRIPT",
    os.path.join(mcore_path, "pretrain_gpt.py"),
)

if not os.path.isfile(pretrain_script):
    raise FileNotFoundError(f"pretrain_gpt.py not found: {pretrain_script}")

sys.argv[0] = pretrain_script
runpy.run_path(pretrain_script, run_name="__main__")
