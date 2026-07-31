# Standalone TE TN GM6 wgrad

本文记录当前仓库中的 GM6 接入、启动方式和已知限制。GM6 是 MoE expert
weight-gradient 的可选路径；它不替换 MATE 的 fprop/dgrad，也不改变普通
Transformer Engine grouped GEMM 的默认路径。

## 上游来源和版本边界

本分支不是基于 `caizhi-mt/llm_pretrain_script#2` 的当前 HEAD，而是基于该 PR 的
历史节点 `ae9f044f355890e9bc769c7b2d0207140fb3a6ab` 做选择性迁移。该节点包含：

1. `33128cc`：MATE fprop/dgrad + TE grouped wgrad；
2. `5ed1bf7`：尝试 pinned/non-blocking DeepEP counts；
3. `6199914`：因 GroupGEMM 退化，完整回退上述 counts 实验；
4. `ae9f044`：MUBIN metadata cache、deferred counts 和 CPU affinity。

PR#2 后续的 `c24cf0d`（MLA RoPE/QKV 布局融合）和 `96b6e85`（Agent 文档）不在
本分支范围内。当前分支在 `ae9f044` 的 MATE/DeepEP 基础上增加独立 GM6 wgrad、
仓库内依赖路径、router-fusion launcher、有限步验证所需的启动变量修复，以及
`SWEEP_SKIP_MOE_METRICS` 兼容处理。

## 代码路径

- `musa_patch/tn_gm6/tn_gm6.cpp`：C++/MUSA 扩展，调用 muDNN routed
  `AsmKernelTCEGroupGemm`，计算 `grad_output.T @ input`，输入 BF16、输出 FP32。
- `musa_patch/tn_gm6/_tn_gm6.so`：当前集群 ABI 下已构建的扩展。训练运行时不需要
  muDNN 源码树、sparse-MoE checkout 或生成 kernel library，但仍需要节点上的
  PyTorch-MUSA、Transformer Engine 和对应 MUSA 运行时库。
- `musa_patch/tn_gm6/loader.py`：延迟加载及 Python 调用封装。
- `musa_patch/te_tn_gm6.py`：包装 TE 的三个 `general_grouped_gemm` Python 入口，
  命中条件时转到 GM6，否则原样调用 TE。
- `musa_patch/mate_grouped_gemm.py`：MATE 路径下可选地直接调用同一个 GM6 wgrad。
- `musa_patch/__init__.py`：通过环境变量预加载扩展并安装 wrapper。预加载很重要，
  因为 GM6 和 TE 可能注册重叠的 ASM dispatcher symbol。

## GM6 改动细节

### 独立 C++/MUSA 扩展

`tn_gm6.cpp` 接收 packed 的 `grad_output`、`input`、CPU INT64 expert sizes 和连续
FP32 输出，逐 expert 构造 variable-K `MatMulParam`，最终调用 muDNN routed
`AsmKernelTCEGroupGemm`。数学语义是：

```text
weight_grad[e] = grad_output[e].T @ input[e]
```

输入是 BF16，输出和累加缓冲是 FP32。`accumulate=false/true` 分别映射 beta=0/1；
因此首个 microbatch 覆写 `main_grad`，后续 microbatch 原位累加。扩展会检查设备、
dtype、二维 packed 输入、三维连续输出、group 数量和 group sizes 总和，失败时直接
报错，不静默产生错误结果。

当前 `_tn_gm6.so` 包含针对真实 FC1/FC2 形状调优的 persistent GM6 dispatcher/kernel。
persistent grid 的目标是让固定 grid 持续领取多个 expert GEMM 工作，减少小/不均匀
expert 的反复 launch 和调度开销；它不是任意 shape 都必然更快，所以 Python wrapper
只在严格命中条件下进入 GM6，其他情况保持 TE fallback。

### Transformer Engine 接入

`te_tn_gm6.py` 同时替换 TE 的 `cpp_extensions`、`gemm` module 和
`grouped_linear` 中已绑定的三个 `general_grouped_gemm` 引用，避免只 patch 一个
Python symbol、实际 `GroupedLinear` 仍调用旧引用。它只拦截
`layout="NT" && grad=True` 的 BF16×BF16→FP32 grouped wgrad。

空 expert 不直接传入底层 GM6：wrapper 把连续非空 expert 切成 active runs；run 少于
两个 group、输入/输出不是同一 packed storage 或其他条件不满足时，该 run 回退 TE。
beta=0 时空 expert 输出清零，beta=1 时保持原 `main_grad`，与原 TE 累加语义一致。

### MATE + main_grad 接入

`mate_grouped_gemm.py` 保留 PR#2 的 MATE fprop/dgrad，并把 expert 参数在 DDP 接管前
打包为连续 storage。wgrad 默认仍调用一次 TE `general_grouped_gemm`，直接写每个
expert 的 FP32 `weight.main_grad`，再设置 `grad_added_to_main_grad`，防止 Megatron
DDP 重复累加。开启 `TE_TN_GM6_WGRAD=1` 后，这次 TE wgrad 调用由上述 wrapper
接入 GM6；这就是当前 16 机成功配置。

另有 `MATE_TN_GM6_WGRAD=1` 的 MATE backward 直连模式，但它要求所有 expert 非空、
packed main_grad 且 `MATE_USE_MAIN_GRAD=1`。当前稳定配置保持它为 0，优先使用具备
active-run/空 expert fallback 的 TE wrapper。

## 依赖

### 16 机验证环境

2026-07-31 的成功运行使用以下环境；`_tn_gm6.so` 是 ABI 相关二进制，不应默认认为
其他 Python、PyTorch-MUSA、MUSA 或设备版本兼容：

| 组件 | 验证版本 |
| --- | --- |
| Python | 3.10.12 |
| PyTorch | 2.5.0 |
| torch_musa | 2.5.0+0324c66 |
| Transformer Engine | 2.0.0 |
| MUSA driver/runtime | 4.3 / 4.3 |
| 设备 | MTT S5000，8 卡/节点 |
| muDNN runtime | 3.1.5，`release_musa_4.3.0`，commit `4ea959f` |
| mate / mate-mubin | 0.2.5 / 0.2.5 |
| tilelang_musa | 0.1.8+musa.3.gitc3ed1bd5 |
| TVM | 0.23.dev0（随 tilelang） |
| DeepEP Python module | `deep_ep`，集群预装 |

`ldd _tn_gm6.so` 在该环境中直接依赖 PyTorch/torch_musa、`libmusart.so.4`、
`libmudnn.so.3`、`libmudnn_xmma.so`、`libmudnn_ops.so`、`libmudnn_tensor.so`、
`libmccl.so.2` 等集群运行库。虽然训练不需要 muDNN 源码树，仍必须有 ABI 匹配的
muDNN runtime。

### 构建时依赖

重新构建 `_tn_gm6.so` 还需要：

- muDNN 源码头文件（含 internal xmma/routed GroupGEMM 接口）；
- routed dispatcher object；
- GM6 kernel library，必要时再提供 NN auxiliary kernel library；
- MUSA compiler、torch_musa extension headers 和与目标节点一致的 Python ABI；
- 本次构建使用的 MUSA ASM 包来自
  `release_musa_4.3.0/2026-01-26/musa_asm.tar.gz`。

这些是 build-time 输入，不应通过 `LD_LIBRARY_PATH` 指向另一个临时 checkout 来
“修复”运行时导入。运行时应只加载仓库内 `_tn_gm6.so` 和系统 ABI 匹配库。

## 必须遵守的导入顺序

`musa_patch.tn_gm6._tn_gm6` 必须先于任何 `transformer_engine` 模块加载。这不是普通的
Python import 风格问题：GM6 与 TE 包含重叠的 ASM dispatcher symbol。如果 TE 先加载，
随后在第一次 wgrad 中懒加载 GM6，GM6 kernel 可能解析到 TE 已驻留的 dispatcher，造成
“单 kernel 测试能跑、完整模型路径报错或行为异常”的差异。

因此 `musa_patch/__init__.py` 会在 `patch_before_import_megatron()` 之前检查
`TE_TN_GM6_WGRAD` 和 `MATE_TN_GM6_WGRAD`；任一个为 1 都立即 preload `_tn_gm6.so`。
不要把这个加载移动到 `te_tn_gm6.py` wrapper、MATE backward 或第一次 GEMM 调用里。
训练入口也必须先 `import musa_patch`，再导入 Megatron/Transformer Engine；仓库现有
Megatron 启动方式满足该顺序。

## 三种启用方式

### TE grouped wgrad（推荐作为独立验证）

```bash
export MATE_GROUPED_GEMM=0
export TE_TN_GM6_WGRAD=1
```

这条路径直接拦截 TE `general_grouped_gemm(layout="NT", grad=True)`，不依赖 MATE。
未命中 GM6 条件的调用会回退到 TE，因此可以先用短步数 smoke test 验证加载和数值。

### MATE fprop/dgrad + TE wrapper GM6 wgrad（当前 16 机稳定配置）

```bash
export MATE_GROUPED_GEMM=1
export MATE_USE_MAIN_GRAD=1
export TE_TN_GM6_WGRAD=1
export MATE_TN_GM6_WGRAD=0
```

MATE 负责 fprop/dgrad；MATE backward 仍发出一次 TE
`general_grouped_gemm(layout="NT")`，再由 `te_tn_gm6.py` 按 active runs 转入 GM6。
该模式保留空 expert fallback，是当前验证和推荐方式。

### MATE backward 直连 GM6（实验模式）

```bash
export MATE_GROUPED_GEMM=1
export MATE_USE_MAIN_GRAD=1
export MATE_TN_GM6_WGRAD=1
export TE_TN_GM6_WGRAD=0
```

此时 MATE 负责 fprop/dgrad，并直接把 GM6 wgrad 写入每个 expert 的 FP32
`weight.main_grad`。要求所有节点上的 `mate`/`mate-mubin` 版本匹配；GM6 本身仍是
独立扩展。若某个 batch 有空 expert，MATE 直连 GM6 会放弃该调用并回退 TE。

## 命中条件

GM6 只接受：

- `layout="NT"`、`grad=True`，且不是 GELU/bias/single-output/D_dtype 特殊路径；
- A/B 为 MUSA BF16、输出为连续 FP32；
- 每个 active expert 的输入和 grad-output 是同一块连续 packed storage；
- 至少两个 expert，形状满足各组相同 N/K；
- 未开启 `ENABLE_ZERO_BUBBLE=1`。

`accumulate` 会映射到 GM6 的 beta：首个 microbatch 使用 beta=0，后续累加使用
beta=1。使用 `main_grad` 时，调用方会把 `grad_added_to_main_grad` 标记为已写入，
避免 Megatron DDP 再加一次梯度。

## 16 机启动示例

2026-07-31 已用本地 Git 提交 `bb587447dd6e980800db459f0d020eadf964f2fa`
的原样归档完成 16 机 5-step 验证。源码归档 SHA-256 为
`1fd20074b969b961a21716f28b9c496d5a8b788ce58cd885b87f0fcbf07f1108`；
训练后 `tar --compare` 无差异，实际启动链没有 CRLF。

验证拓扑和训练参数：

- 16 节点 × 8 卡 = 128 ranks，TP/PP/EP=`2/8/8`；
- 24 层，pipeline first/last 都是 3 层；
- MoE layout 为 `([0]*3+[1]*21)`，不要替换成其他实验 layout；
- 256 experts、top-k 8、seq=4096、MBS=1、GBS=2048；
- flex dispatcher + DeepEP + router fusion；
- profiler 关闭，`TRAINING_STEPS=5`。

成功配置使用 MATE fprop/dgrad、TE wrapper 接 GM6 wgrad：

```bash
export USE_DEEPEP_ACE=1
export MOE_GROUPED_GEMM=1
export MATE_GROUPED_GEMM=1
export MATE_USE_MAIN_GRAD=1
export MATE_CACHE_MUBIN_DISPATCH=1
export MATE_DEFER_DEEPEP_COUNTS=1
export MUSA_CPU_AFFINITY=1
export MUSA_CPU_AFFINITY_MODE=mate
export MUSA_CPU_AFFINITY_MAP='0-7;8-15;16-23;24-31;64-71;72-79;80-87;88-95'
export TE_TN_GM6_WGRAD=1
export MATE_TN_GM6_WGRAD=0
export ENABLE_PROFILER=0
```

注意这里同时开启 `MATE_GROUPED_GEMM` 和 `TE_TN_GM6_WGRAD`：MATE 处理
fprop/dgrad，MATE backward 的 wgrad 调用 TE `general_grouped_gemm`，再由 TE wrapper
转入 GM6。不要把它误写为 `MATE_TN_GM6_WGRAD=1`。

Router fusion 必须保持开启。训练入口使用仓库内
`llm_pretrain_script/pretrain_gpt_musa_routerfusion_launcher.py`，它先导入
`musa_patch`，再补注册 `--moe-router-fusion` 并注入当前 MUSA TE 已有的 fused-router
接口，以绕过只按 TE 版本号判断导致的误禁用。不能退回普通
`pretrain_gpt_musa_launcher.py`。

本次验证没有把 16 机缩层测试脚本提交进仓库；它作为仓库外 harness 保留原 ws128
脚本，只有代码路径、hostfile、输出路径和 5-step 测试配置不同。实际执行命令为：

```bash
cd /home/jd/haowen.yan
bash /home/jd/haowen.yan/verify_bb58744_harness/launch.sh
```

`launch.sh` 的等价核心命令如下，使用提交内的有限运行 runner，不使用会在正常完成后
重启任务的 daemon manager：

```bash
CODE_ROOT=/home/jd/haowen.yan/verify_bb58744_exact/llm_pretrain_script
WORK_DIR=${CODE_ROOT}/llm_pretrain_script
HARNESS_ROOT=/home/jd/haowen.yan/verify_bb58744_harness
RUN_ROOT=/home/jd/haowen.yan/training_runs
export LOG_NAME=verify_bb58744_exact_steps5_$(date +%Y%m%d_%H%M%S)
export LLM_PRETRAIN_WORK_DIR=${WORK_DIR}
export LLM_PRETRAIN_DIST_TRAIN=scripts/dist_train_megatron.sh
export MUSA_PRETRAIN_ENTRY=../../../verify_bb58744_harness/entry.sh
export TRAINING_STEPS=5

bash ${WORK_DIR}/cluster/dist_run_megatron.sh \
  ${HARNESS_ROOT}/hostfile \
  --logdir ${RUN_ROOT}/manager_logs/${LOG_NAME} \
  --output-dir ${RUN_ROOT}/outputs/${LOG_NAME}
```

外部 `entry.sh` 只设置仓库内 tokenizer/Megatron/patch 路径、数据路径和工作目录下的
输出路径，然后执行含上述 24 层参数的 `train_config.sh`。hostfile 对应
`hostfile.runtime.jd_llmtest_free_mccl_good16_20260727` 的 16 个可用节点。

成功 run 为 `verify_bb58744_exact_steps5_20260731_222217`。step 1 warmup 为
110.538 s；step 2–5 平均 76.190 s/step、98.05 TFLOP/s/GPU；loss 从
11.89896 降至 11.43728，skipped/NaN 均为 0，iteration 5 checkpoint 保存成功，
16 节点无残留进程。

## 历史脏修改和最终修复

早期 GM6 开发直接发生在远端可运行 worktree 中；该目录同时含未提交源码修改、测试
launcher/hostfile、日志、core dump、编译产物和换行噪声。第一次把它同步回本地时，
“远端能跑”与“本地 Git 提交”并不等价：旧独立验证目录相对重新 clone 的仓库曾出现
1372 个原始文件哈希差异，其中也包含运行路径文件，不能作为提交验证证据。

随后执行了以下修复：

1. 删除被污染的本地 checkout，重新 clone `arcing-mt/llm_pretrain_script`；
2. 以 `ae9f044` 为上游历史基点，只迁移 MATE/DeepEP/affinity 正常源码修改；
3. GM6 源码、预构建扩展、router-fusion launcher 和仓库内路径单独纳入 Git；
4. 不迁移未跟踪日志、测试启动脚本、hostfile、core dump 和临时 benchmark；
5. `f5bea26`/`a84ccef` 处理过 Windows CRLF 噪声；最终验证不再为换行创建新提交，
   而是直接用 Git archive，并在远端启动前检查实际运行脚本无 `\r`；
6. `bb58744` 补齐 deferred DeepEP counts、CPU affinity/env 透传、
   `SWEEP_SKIP_MOE_METRICS` 和单测后，从该提交重新生成干净归档验证。

因此应以最终 Git tree 和上述 archive/hash/5-step 结果为准，不应以早期远端 worktree、
中间验证目录或单个历史 commit 的“看起来能跑”作为交付依据。

## 构建和回退

当前 `_tn_gm6.so` 是集群 ABI 的预构建产物。若 Python、PyTorch-MUSA、设备型号或
MUSA/muDNN ABI 改变，需在 MUSA 节点重新构建，不能在普通本地 Windows 环境编译：

```bash
cd megatron-lm-musa-patch/musa_patch/tn_gm6
export TE_TN_GM6_MUDNN_SOURCE_ROOT=/path/to/mudnn
export TE_TN_GM6_DISPATCHER_OBJ=/path/to/dispatcher.o
export TE_TN_GM6_KERNEL_LIB=/path/to/libkernel.so
python setup.py build_ext --inplace
```

生产回退只需取消变量或设为 0：

```bash
export TE_TN_GM6_WGRAD=0
export MATE_TN_GM6_WGRAD=0
```

此时 MATE（若开启）仍可负责 fprop/dgrad，wgrad 回到 Transformer Engine
`general_grouped_gemm`；若同时关闭 MATE，则整个 expert 计算回到原始 TE 路径。

## 排查顺序

1. 先确认每个节点能 `import musa_patch.tn_gm6._tn_gm6`，并确认 Python/PyTorch-MUSA
   ABI 一致。
2. 用 2 step、单一配置验证数值和 `[TE_TN_GM6]` 日志，再扩到 16 机。
3. 若没有 GM6 日志，先检查环境变量是否被 launcher 透传，再检查 layout、dtype、
   packed storage 和空 expert 条件；未命中时是预期的 TE fallback，不等于训练卡死。
4. 性能比较必须使用相同 hostfile、batch、层数、dispatcher、MATE 状态和 profiler
   区间；不要把 profiler 开销与无 profiler 运行直接比较。
