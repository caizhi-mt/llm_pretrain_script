"""Remote-only build script for the standalone BF16 TN GM6 extension."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension


SOURCE_DIR = Path(__file__).resolve().parent
MUDNN_ROOT = Path(os.environ["TE_TN_GM6_MUDNN_SOURCE_ROOT"]).resolve()
DISPATCHER = Path(os.environ["TE_TN_GM6_DISPATCHER_OBJ"]).resolve()
KERNEL_LIB = Path(os.environ["TE_TN_GM6_KERNEL_LIB"]).resolve()
AUX_KERNEL_LIB_NN = os.environ.get("TE_TN_GM6_AUX_KERNEL_LIB_NN")
if AUX_KERNEL_LIB_NN:
    AUX_KERNEL_LIB_NN = Path(AUX_KERNEL_LIB_NN).resolve()

for path in (DISPATCHER, KERNEL_LIB, AUX_KERNEL_LIB_NN):
    if path is None:
        continue
    if not path.is_file():
        raise FileNotFoundError(path)

extension = MUSAExtension(
    name="_tn_gm6",
    sources=[
        str(SOURCE_DIR / "tn_gm6.cpp"),
        str(SOURCE_DIR / "musa_extension_stub.mu"),
    ],
    include_dirs=[
        str(MUDNN_ROOT / "include"),
        str(MUDNN_ROOT / "src/internal/kernels"),
        str(MUDNN_ROOT / "src/internal/kernels/xmma"),
    ],
    define_macros=[("MUSA_ROUTED_GROUP_ASM", "1")],
    extra_objects=[
        str(DISPATCHER),
        *([str(AUX_KERNEL_LIB_NN)] if AUX_KERNEL_LIB_NN else []),
        str(KERNEL_LIB),
    ],
    extra_compile_args={"cxx": ["-O3"], "mcc": ["-O3"]},
    extra_link_args=[f"-Wl,-rpath,{KERNEL_LIB.parent}"],
)

setup(
    name="te-tn-gm6-extension",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
