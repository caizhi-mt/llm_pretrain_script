# Agent 维护指南与功能变更记录

本文档面向后续在本仓库工作的 agent。开始修改前先阅读本文档和 `llm_pretrain_script/README.md`，避免重复已经回退的实验或破坏当前默认快路径。

## 强制维护规则

每次修改训练行为、性能路径、环境变量、算子实现或默认配置时，必须在同一个 PR 中同步更新本文档：

1. 在“按提交记录”末尾新增一项，写清日期、commit/主题、修改文件、解决的问题和最终功能。
2. 写明默认是否启用、适用条件、回退开关，以及不满足条件时的行为。
3. 记录实际执行的精度、性能和单测验证；没有验证的项目必须明确标注，不能推断为已通过。
4. 被回退的方案也要保留记录并说明原因，后续不要在没有新证据时原样恢复。
5. 日志、trace、checkpoint、临时 benchmark 和凭据不得提交。性能数字必须注明测试拓扑，不能直接外推到 128 机生产配置。

## 当前默认功能矩阵

| 功能 | 默认值 | 功能与回退 |
|---|---:|---|
| `MOE_GROUPED_GEMM` | `1` | 使用 GroupedMLP；设为 `0` 回退 SequentialMLP。 |
| `MATE_GROUPED_GEMM` | `1` | MoE expert fprop/dgrad 使用 MATE，wgrad 使用 TE；设为 `0` 完整回退 TE GroupedLinear。 |
| `MATE_USE_MAIN_GRAD` | `1` | TE wgrad 直写 FP32 `main_grad`；设为 `0` 返回临时 weight grad。 |
| `MATE_FLASH_ATTN` | `1` | BF16 MLA FA forward 使用 MATE MUBIN，backward 保持原生 MUSA；设为 `0` 完整回退原 TE FA。 |
| `MATE_CACHE_MUBIN_DISPATCH` | `1` | 缓存 GroupGEMM/FA 不可变 MUBIN module、dispatcher、最终 `GemmMubinId` kernel path 与 launch handle；不缓存 tensor、路由 counts、LSE 或算子输出。 |
| `MATE_DEFER_DEEPEP_COUNTS` | `1` | 延迟构造 device counts，并从 device routing map 归约得到；设为 `0` 回退 dispatch 后立即构造 device tensor。 |
| `MUSA_COMPACT_PERMUTE` | `1` | DeepEP local permute/unpermute 使用 `[tokens, router_topk]` compact row map 调用 TE MUSA kernel；设为 `0` 回退 `[tokens, local_experts]` dense TE 路径。 |
| `MUSA_FUSED_MLA_DOWN_PROJ` | `1` | TP1、无 bias 的 MLA q/kv down projection 合并 forward/dgrad GEMM，同时保留原 checkpoint 参数和独立 FP32 `main_grad`；设为 `0` 回退两个独立 Linear。 |
| `MUSA_CPU_AFFINITY` | `0` | 默认不绑核；设为 `1` 后按 `MUSA_CPU_AFFINITY_MODE/MAP` 绑核。推荐 `mode=mate`。 |
| `MUSA_NATIVE_ROPE` | `1` | 标准 RoPE 使用 MUDNN `torch.rope`；设为 `0` 回退 eager 组合算子，并同时禁用下面的 MLA 布局融合。 |
| `MUSA_FUSED_MLA_ROPE` | `1` | 标准 MLA RoPE 与 Q/K/V 布局融合；设为 `0` 只保留 MUDNN `torch.rope`。 |

当前 ws128 模型仍传入 `--no-rope-fusion`，但 `MUSA_FUSED_MLA_ROPE` 是独立的 MUSA 标准 RoPE 快路径，不受该参数关闭。它仅在标准 RoPE、MUSA BF16/FP16、CP=1、非 packed、非 inference、相关 head dim 为 2 的幂时生效。

## 按提交记录

### 2026-07-29 — `33128cc`：接入 MATE fprop/dgrad 与 TE wgrad

- 主要文件：
  - `megatron-lm-musa-patch/musa_patch/mate_grouped_gemm.py`
  - `Megatron-LM/megatron/core/transformer/moe/experts.py`
  - `Megatron-LM/megatron/core/transformer/moe/fused_a2a.py`
  - `megatron-lm-musa-patch/musa_patch/__init__.py`
  - ws128 启动脚本与 `test_mate_grouped_gemm.py`
- 修改内容：保留 TE `GroupedLinear` 参数与 state-dict 结构，fprop/dgrad 调用 MATE `ragged_m_moe_gemm_16bit`，wgrad 通过一次 TE `general_grouped_gemm(layout="NT")` 执行。
- 功能：wgrad 默认直接累积到持久 FP32 `main_grad`，避免 BF16 临时梯度及后续 BF16→FP32 add/cast；利用 `grad_added_to_main_grad` 区分首个和后续 microbatch。
- 元数据：DeepEP host splits 保存在 `_mate_m_splits`，MATE 使用同设备连续 INT32 counts，TE wgrad 继续使用 Python split list，避免 expert 层 `tolist()` 同步。
- 约束：仅 BF16、连续 MUSA tensor、无 bias、非 FP8、packed expert weights；不满足条件时打印一次 fallback 并走原 TE。
- 启动检查：所有节点必须安装同版本 `mate` 和 `mate-mubin`。

### 2026-07-29 — `5ed1bf7`：尝试 pinned counts 非阻塞 H2D

- 主要文件：`Megatron-LM/megatron/core/transformer/moe/fused_a2a.py`。
- 修改内容：先在 pinned CPU tensor 中构造 DeepEP counts，再以 `non_blocking=True` 复制到 device，并保持 host tensor 生命周期。
- 目标功能：减少 DeepEP dispatch 后 counts H2D 对 CPU/GPU 关键路径的阻塞。
- 最终结论：该方案虽然减少了表面空泡，但与 GroupGEMM 竞争后使其耗时变长，下一提交已完整回退。后续不要原样恢复。

### 2026-07-29 — `6199914`：回退 pinned counts 非阻塞 H2D

- 主要文件：`Megatron-LM/megatron/core/transformer/moe/fused_a2a.py`。
- 修改内容：移除 pinned host counts、异步 `.to()` 和额外生命周期引用，恢复直接 device tensor 构造。
- 原因：trace 显示前一方案导致 GroupGEMM 性能下降；空泡收益不能覆盖 GEMM 回退。
- 功能影响：只撤销 `5ed1bf7` 的实验，不撤销 MATE/TE 混合 GroupedLinear。

### 2026-07-30 — `ae9f044`：优化 MATE 调度、DeepEP counts 与 CPU 绑核

- 主要文件：
  - `Megatron-LM/megatron/core/transformer/moe/fused_a2a.py`
  - `megatron-lm-musa-patch/musa_patch/deepep_ace/token_dispatcher.py`
  - `megatron-lm-musa-patch/musa_patch/mate_grouped_gemm.py`
  - `megatron-lm-musa-patch/musa_patch/cpu_affinity.py`
  - `megatron-lm-musa-patch/docs/mate_cpu_affinity.md`
  - ws128 分发/启动脚本及聚焦单测
- Deferred counts：DeepEP dispatch 后暂存 CPU INT32 tensor；在 shared-expert GEMM 已提交后，从 device routing map 按 8192 行分块归约 device counts。分块用于规避当前 MUSA 大列 bool reduction 精度问题，同时去掉 pageable H2D 和 `sum().item()`。
- MUBIN cache：缓存 `MoeGemmMubinDispatcher` 和已校验的 kernel artifact；每次仍重新执行 kernel selection，路由变化不会错误命中旧 counts。
- CPU affinity：新增 `early` 和 `mate` 两种模式。推荐 `mate`，只绑定提交 MATE 工作的 Python 线程，DeepEP/MCCL/OpenMP 线程不继承限制；Intel/AMD 绑核方法见独立文档。
- 默认行为：MATE、main-grad、MUBIN cache、deferred counts 改为默认启用；CPU affinity 仍默认关闭。所有变量通过 SSH 白名单透传。
- 验证摘要：单机 8 卡 EP8 trace 中，DeepEP→shared FC1 空泡 `0.872 → 0.109 ms`，permute→FC1 空泡 `0.351 → 0.175 ms`；非 profiler 稳态 step `608.10 → 608.36 ms`，端到端视为持平，未观察到 GEMM 结构性回退。

### 2026-07-30 — `c24cf0d`：标准 MLA RoPE 与 Q/K/V 布局融合

- 主要文件：
  - `megatron-lm-musa-patch/musa_patch/rotary_pos_embedding.py`
  - `Megatron-LM/megatron/core/transformer/multi_latent_attention.py`
  - `Megatron-LM/megatron/core/fusions/fused_mla_yarn_rope_apply.py`
  - ws128 启动脚本、README 和 `test_native_rope.py`
- Native RoPE：标准 `rope` 的 unfused 路径默认调用 MUDNN `torch.rope(..., multi_latent_attention=True)`，替代 eager `cos/sin/mul/cat`。
- MLA layout fusion：复用 MLA Triton kernel，Q 原位完成 RoPE；KV 一次完成 split、K RoPE/broadcast 以及连续 K/V 输出，消除 attention 前 4 次目标 `cat` 和 7 次相关布局拷贝。
- MUSA dQ：只在 MUSA 上把 Q backward 的 head tile 从 2 调为 16；生产 shape 微测约 `0.95 → 0.41 ms`，CUDA 保持原 tile。
- 精度：单机 EP8、1 层、seq4096、MBS2、BF16、fake data 的 4-step loss 最大绝对差 `1.3e-4`，无 NaN/skipped iteration；算子级 Q/K/V 和输入梯度通过 BF16 容差。
- 性能：同配置 8-rank trace 的 Profiler step 中位数 `541.895 → 538.808 ms`（`-0.57%`），GPU active union 均值减少 `3.46 ms`；MATE GroupGEMM/TE wgrad 波动低于 `0.5%`。

### 2026-07-30 — Agent 文档与 PR 维护

- 新增本文件，把当前 PR 的功能、历史实验、回退原因、开关和验证集中交给后续 agent。
- 本项不改变训练运行时行为。

### 2026-08-03 — MATE MLA FA forward 与原生 MUSA backward

- 主要文件：
  - `megatron-lm-musa-patch/musa_patch/mate_flash_attention.py`
  - `megatron-lm-musa-patch/musa_patch/__init__.py`
  - ws128 分发/启动脚本、README 和 `test_mate_flash_attention.py`
- 实现：只替换 TE 模块实际引用的 `flash_attn_func`。BF16 MLA fixed-length forward
  直接调用缓存后的 MATE MUBIN launch；保存 output/LSE 后，backward 继续调用
  `aten::_scaled_dot_product_attention_flash_musa_backward`。不调用 MATE 0.2.5
  自带的 varlen backward。
- 缓存：复用 `MATE_CACHE_MUBIN_DISPATCH`，key 包含 MATE/artifact 根目录、device、
  arch、dtype、causal、QK/V head dim；只缓存 immutable launch 元数据。首次选择在
  profiler warmup 前完成，forward/recompute 均复用同一 launch。
- 默认与回退：`MATE_FLASH_ATTN=1` 默认启用；仅支持 MP31、BF16、Dqk=192、
  Dv=128、causal、dropout=0、CP=1、非 recompute-variance 路径。其他 shape/功能走
  原 TE FA。当前私有 API 固定要求 `mate=mate-mubin=0.2.5`；选中快路径后的
  runtime/kernel 错误 fail-fast，禁止不同 rank 静默分叉。
- 算子精度：seq4096、MBS2 下 TP1/128 heads 与 TP2/64 heads 均验证。output 和
  dQ/dK/dV cosine 不低于 `0.9999928`，max abs 不超过 `0.03125`，无 NaN/Inf。
- 算子性能：TP1 forward 中位数 `4.808 → 4.432 ms`（`-7.8%`）；TP2/64 heads
  `2.384 → 2.247 ms`（`-5.7%`）。native backward 耗时保持在测试波动内。
- 两层训练精度：单机 EP8、seq4096、MBS2、full/uniform recompute、fake data、
  force-load-balancing、hidden-loss 10 step 全部完成。相对 native 的 hidden loss
  差异为 `0.116%–0.164%`，未随 step 增长；router seq loss 每 step 完全一致，
  无 NaN/skipped iteration。max allocated 均为 `64,026 MB`。
- 两层训练性能：稳定 step 4–10 均值 native `718.77 ms`、MATE `721.87 ms`，
  差异 `+0.43%`，小于单次 MoE step 波动，不能宣称端到端收益。Trace 中四次
  FA forward kernel 累计 `18.566 → 17.102 ms`（`-7.89%`），全部
  `rotary_fwd_kv → MATE FA` device gap 小于 `0.003 ms`；四次 forward 加两次
  native backward 的 kernel 总时长 `41.740 → 39.813 ms`（`-1.927 ms`）。

### 2026-08-03 — 完善 MATE GroupGEMM MUBIN 调度缓存

- 主要文件：
  - `megatron-lm-musa-patch/musa_patch/mate_grouped_gemm.py`
  - `megatron-lm-musa-patch/test/test_mate_grouped_gemm.py`
  - `llm_pretrain_script/README.md`
- 实现：在既有 dispatcher 和 artifact verification cache 之上，增加默认 MUBIN
  module 路径缓存，并按最终不可变 `GemmMubinId` 缓存 kernel path。动态 `M` 仍先
  参与 block/ASM id 选择；当前 input、output、weights、routing counts 和实际 launch
  scalar 每次重新传入，不缓存任何 tensor 或 data pointer。
- GroupedLinear 检查：按 module、device 和 `use_main_grad` 缓存首次成功的静态
  packed weight/main-grad layout 检查；input/counts 的 dtype、device、contiguous、
  split 数量和当前总 token 数仍逐次校验。DDP 尚未安装 `main_grad` 时的失败结果不缓存。
- 默认与回退：沿用 `MATE_CACHE_MUBIN_DISPATCH=1`，没有新增环境变量；设为 `0`
  回退 MATE 原生逐次 dispatch。显式 cache dir 或 custom repository 保持原 MATE 语义。
- Trace 验证：单机 8×S5000、EP8、2 层、seq4096、MBS2、BF16、full/uniform
  recompute、fake data、force-load-balancing、无 DeepEP recompute cache。四处
  `permute→FC1` 空泡的每 rank 均值总和 `0.759 → 0.0116 ms`，最大值
  `2.874 → 0.0044 ms`；fprop/dgrad CPU dispatch 分别减少 `58.5%/57.3%`。
- GEMM 验证：MATE fprop+dgrad GPU kernel 总时长 `91.285 → 91.142 ms`
  （`-0.16%`）；TE wgrad `37.064 → 37.185 ms`（`+0.33%`）；全部 GEMM
  `211.725 → 211.874 ms`（`+0.07%`），均在运行波动内，没有结构性回退。
- 训练验证：同环境 30-step cache off/on 稳态均值 `713.592/713.776 ms`，统计持平；
  hidden loss 最大相对差 `0.0061%`，无 NaN/skipped iteration，max allocated 均为
  `64,029 MB`。完整 cache 下原生/MATE MLA 稳态均值 `716.004/713.776 ms`，当前
  两层测试中 MATE MLA 快 `2.228 ms`（`0.31%`）；该数字不可直接线性外推到生产拓扑。
- 单测：`test_mate_grouped_gemm.py` 与 `test_cpu_affinity.py` 共 10 项通过；另以
  MATE 0.2.5 实际 dispatcher 验证不同动态总 `M`、相同 `GemmMubinId` 复用同一
  kernel path。

### 2026-08-03 — DeepEP local Permute/Unpermute compact row map

- 主要文件：
  - `megatron-lm-musa-patch/musa_patch/compact_permutation.py`
  - `megatron-lm-musa-patch/musa_patch/deepep_ace/token_dispatcher.py`
  - `megatron-lm-musa-patch/test/test_compact_permutation.py`
  - ws128 启动脚本与本文档
- 实现：DeepEP 已返回 `[N, router_topk]` indices/probs。仍用当前动态 routing 生成
  TE expert-major row id，但在数据搬运前把 row map 收缩为 `[N, router_topk]` 和
  `[router_topk, N]`，使 TE native MUSA permute/unpermute 只扫描 top-k 列，而不是
  32 个 local-expert 列。unpermute backward 保持 TE 原生转置 map kernel。
- 动态语义：每次 dispatch 都重新生成 row map；不缓存 routing、counts、tensor、
  data pointer 或专家选择，不假设 counts 均衡。无效 `-1` slot 的 probability grad
  由 TE kernel 写零。只长期保留一份 compact 转置映射，token-major 映射仅在
  permute forward 期间存在。沿用 DeepEP/router top-k 的每 token expert id 唯一契约。
- 默认与回退：`MUSA_COMPACT_PERMUTE=1` 默认启用；仅接管 DeepEP-ACE、MUSA BF16
  hidden、FP32 compact probs、连续 int32/int64 indices、hidden dim 为 8 的倍数且
  top-k 为 4 的倍数的路径。
  其他 dtype、shape、设备或显式设为 `0` 时完整回退原 TE dense 实现。
- 算子精度：非均匀 expert counts、含无效 slot 的 4096×2048 测试中，permuted
  hidden/probs、restored hidden、hidden grad 和 compact probs grad 均逐元素一致。
- Trace：单机 8×S5000、EP8、2 层 MoE、seq4096、MBS2、BF16、full/uniform
  recompute、fake data、force-load-balancing。四段 local permutation kernel 的
  8-rank 均值总和 `14.792 → 13.537 ms`，减少 `1.255 ms`（`8.48%`）；其中
  permute backward `3.064 → 2.068 ms`。全部 GEMM `211.874 → 211.734 ms`，无回退。
- 训练性能：两组反向顺序 30-step A/B 的稳态中位数，compact 分别快
  `3.0/2.3 ms`；按每组 MAD 去除系统长尾后的均值合并收益约 `1.76 ms`
  （`0.25%`）。hidden loss 最大相对差 `0.0175%`，无 NaN/skipped iteration；
  该小幅收益仍需在生产拓扑复测。
- 单测：`test_compact_permutation.py` 共 7 项通过，覆盖 raw `uint8` preallocated
  storage 与 expanded/stride-0 backward grad；其中 MUSA 测试需显式设置
  `RUN_MUSA_TESTS=1`。

### 2026-08-03 — Wgrad、MATE MLA backward 与 DeepEP 长尾审计

- 专用 ragged wgrad 原型：实现了直接消费 packed input/grad、动态非均匀 counts、
  直接输出 FP32 `[experts, N, K]` 的单-launch Triton kernel，数值与 TE 逐元素一致。
  生产型 `E128, M≈64, N=1536, K=2048` 为 `14.49 ms`，TE 为 `2.70 ms`；当前
  两层测试的 `E32, M≈2K, N=7168, K=2048` 为 `513.7 ms`，TE 为 `6.22 ms`。
  MATE K-contig BF16 临时输出在前一 shape 中为 `3.70 ms` kernel/zero 加 `3.45 ms`
  transpose/cast，仍慢于 TE。原型未接入；继续优化必须新增 MUBIN/MUTLASS 专用
  `[M,K] × [M,N] → FP32 [N,K]` epilogue，不能采用当前 Triton/MATE 临时转置路径。
- MATE MLA backward：MATE 0.2.5 只支持统一 head dim 128/256，而当前 MLA 是
  `Dqk=192, Dv=128`。补零到 256 的 backward 精度可接受（完整 shape dQ/dK 最大
  绝对差 `0.00390625`，dV 一致），但 split plan `45.3 ms`，原生 MUSA backward
  `15.5 ms`；deterministic separate plan 触发 kernel timeout。因此继续保留现有
  MATE forward + native MUSA backward，不提交负优化。
- DeepEP 长尾：当前内部 wheel 为 `deep_ep 1.1.0+9a0d761`，ACE Python API
  `dispatch_ace/combine_ace` 不接受 config，因此 `--moe-deepep-num-sms` 20/40/60
  对 ACE 无实际影响。ACE buffer 1→2、4-core/避让 housekeeping 绑核均无稳定收益；
  标准 DeepEP 30-step 由约 `711.9` 退到 `749.1 ms`。长等待会在 EP peers 之间
  转移，属于 peer-arrival/D2D contention；根治需要修改 DeepEP ACE 二进制并用完整
  EP8 traces 配对发送/接收 rank，Python cache 或 rank-local barrier 不能缩短关键路径。

### 2026-08-03 — MLA q/kv down-projection fusion

- 主要文件：
  - `megatron-lm-musa-patch/musa_patch/fused_mla_down_projection.py`
  - `Megatron-LM/megatron/core/transformer/multi_latent_attention.py`
  - `megatron-lm-musa-patch/test/test_fused_mla_down_projection.py`
  - ws128 启动脚本与本文档
- 实现：q-lora down projection 和 kv-lora/rope down projection 共享相同 hidden input。
  forward 临时 pack 两个 weight 后只下发一个 GEMM；backward 把 q/kv output grad pack 后
  只下发一个 dgrad GEMM。两个原始 Parameter、checkpoint key 和 optimizer state 均不变；
  wgrad 仍分别写入原 q/kv FP32 `main_grad`，不生成额外 FP32 packed grad。
- 默认与回退：`MUSA_FUSED_MLA_DOWN_PROJ=1` 默认启用；仅接管 MUSA FP16/BF16、
  `q_lora_rank != None`、TP1、无 linear bias 且连续 weight 的路径。TP>1、bias、其他
  dtype/device 或显式设为 `0` 时自动回退原两个 Linear。
- 算子性能，`tokens=8192, H=7168, q_rank=1536, kv+rope=576`：forward prepacked
  `0.743 → 0.674 ms`，dgrad `0.952 → 0.625 ms`；wgrad 一个大 GEMM与两个原 GEMM
  均约 `0.710 ms`，因此保留两个独立 main_grad wgrad。
- 两层训练：单机 8×S5000、seq4096、MBS2、BF16、EP8、full recompute、fake data、
  force-load-balancing。三组正反顺序 30-step A/B 的稳态中位数平均减少约 `0.87 ms`，
  MAD 过滤均值平均减少约 `1.28 ms`；trace 中全部 GEMM 8-rank 均值减少 `0.45 ms`，
  copy/cast 增加约 `0.14 ms`。max allocated `64003 → 64020 MB`。
- 精度：三组训练 hidden loss 最大相对差 `0.016%`，无 NaN/skipped iteration；MUSA
  单测覆盖 forward、fused dgrad、两个独立 FP32 main_grad wgrad，共 3 项通过。

## 当前已知边界

- 单机 EP8 结果不能直接代表 ws128 的 TP2/PP8/EP64 生产拓扑；进入长训前仍需同配置短程 A/B。
- 当前 MATE main-grad 路径基于生产配置未开启 overlap-grad-reduce；若未来开启，需要重新验证梯度 ready 语义。
- CPU affinity 与 CPU NUMA/SNC/NPS 和容器 cpuset 强相关，必须按 `mate_cpu_affinity.md` 生成映射；错误映射会 fail-fast。
- MLA RoPE 精度目前验证了算子前后向和 4-step loss，尚未做长程收敛对比。
- 任何优化都必须同时观察空泡与 GEMM kernel 时间；只缩短 CPU 区间但拖慢 GroupGEMM 的方案不能保留。
- compact permutation 当前仍通过 dense bool routing map 生成 expert-major row id；若继续优化该段，优先评估直接从 compact indices 构造 row id，但必须保留非均匀 counts、无效 slot 和 probability-gradient 语义。
