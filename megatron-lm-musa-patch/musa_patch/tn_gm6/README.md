# Standalone TE TN GM6 wgrad extension

`_tn_gm6.so` is intentionally checked in. It contains the persistent BF16
TN GM6 dispatcher and device kernel used by `TE_TN_GM6_WGRAD=1`, so training
does not need a separate muDNN source tree, sparse-MoE checkout, kernel build
directory, or generated kernel library at runtime.

The binary still requires the normal MUSA training runtime already required by
PyTorch MUSA and Transformer Engine. It is built for the cluster's Python 3.10,
PyTorch MUSA ABI, and MTT S5000 target. Rebuild it with `setup.py` when any of
those ABI or device requirements change.

The optional build script is retained for development. Its environment
variables and external source inputs are build-time dependencies only; they
are not read when importing the checked-in extension or starting training.
