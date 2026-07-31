#!/usr/bin/env python3
"""Launch Megatron-LM pretrain_gpt.py with musa_patch applied first."""

import os
import runpy
import sys

if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
    import musa_patch  # noqa: F401 — 必须最先导入，不可提前 import megatron
    import torch

    # musa_patch replaces Megatron's complete MoE argument group with an older
    # copy that predates --moe-router-fusion. Add only that missing option for
    # this experimental launcher; the original patch and launcher stay intact.
    import megatron.training.arguments as megatron_arguments

    _orig_add_moe_args = megatron_arguments._add_moe_args

    def _add_moe_args_with_router_fusion(parser):
        parser = _orig_add_moe_args(parser)
        group = parser.add_argument_group(title="moe router fusion")
        group.add_argument(
            "--moe-router-fusion",
            action="store_true",
            help="Enable TransformerEngine fused MoE router operations.",
        )
        return parser

    megatron_arguments._add_moe_args = _add_moe_args_with_router_fusion

    # This MUSA TE branch provides the fused router APIs while retaining the
    # upstream package version 2.0.0. Megatron gates their imports on TE 2.7,
    # so expose the verified local implementations to the two modules that
    # cached None during import.
    from transformer_engine.pytorch.router import (
        fused_compute_score_for_moe_aux_loss,
        fused_moe_aux_loss,
        fused_topk_with_score_function,
    )
    import megatron.core.extensions.transformer_engine as megatron_te
    import megatron.core.transformer.moe.moe_utils as moe_utils

    for module in (megatron_te, moe_utils):
        module.fused_compute_score_for_moe_aux_loss = fused_compute_score_for_moe_aux_loss
        module.fused_moe_aux_loss = fused_moe_aux_loss
        module.fused_topk_with_score_function = fused_topk_with_score_function
    moe_utils.HAVE_TE = True

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

mcore_path = os.environ.get("MCORE_PATH", "/home/Megatron-LM")
pretrain_script = os.environ.get(
    "PRETRAIN_SCRIPT",
    os.path.join(mcore_path, "pretrain_gpt.py"),
)

if not os.path.isfile(pretrain_script):
    raise FileNotFoundError(f"pretrain_gpt.py not found: {pretrain_script}")

sys.argv[0] = pretrain_script
runpy.run_path(pretrain_script, run_name="__main__")
