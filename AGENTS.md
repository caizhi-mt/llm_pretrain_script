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
| `MATE_CACHE_MUBIN_DISPATCH` | `1` | 缓存不可变 MUBIN dispatcher/artifact 校验元数据；不缓存 tensor、路由 counts 或 kernel selection。 |
| `MATE_DEFER_DEEPEP_COUNTS` | `1` | 延迟构造 device counts，并从 device routing map 归约得到；设为 `0` 回退 dispatch 后立即构造 device tensor。 |
| `DEEPEP_CACHE_RECOMPUTE_DISPATCH` | `1` | BF16 DeepEP-ACE activation recompute 默认复用原 forward 的 dispatch handle；设为 `0` 完整回退每次重新计算 layout/dispatch metadata。无重计算、非 ACE 或 FP8 时不生效。 |
| `DEEPEP_CACHE_CAPTURE_AFTER_FC1` | `1` | TE GroupedMLP 默认在 FC1 GroupGEMM 下发后，将 cache capture clone 排到同一计算流，避免独立 stream 与 matmul 争抢带宽；设为 `0` 回退原独立 cache stream。legacy/非 TE grouped GEMM 自动使用原路径。 |
| `DEEPEP_CACHE_RECOMPUTE_VALIDATE` | `0` | 设为 `1` 时逐次校验 source indices/probs，并通过 EP all-reduce 统一 fallback；仅用于精度调试，不能用于性能测试。 |
| `DEEPEP_CACHE_RECOMPUTE_MAX_ENTRIES` | `256` | 每层允许的最大在途 checkpoint cache entry 数；超限 fail-fast，防止异常或未执行 backward 时无限占用显存。 |
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

### 2026-08-03 — DeepEP activation-recompute handle cache 与 post-FC1 capture

- 主要文件：
  - `Megatron-LM/megatron/core/tensor_parallel/random.py`
  - `Megatron-LM/megatron/core/transformer/moe/moe_utils.py`
  - `megatron-lm-musa-patch/musa_patch/deepep_ace/token_dispatcher.py`
- 生命周期：Megatron checkpoint body 暴露唯一 `checkpoint_id` 和 `forward/recompute` phase。每个 `_DeepepManager` 按 checkpoint ID 保存记录，避免 PP warmup、多 microbatch 和反序 backward 错配；eval/no-grad 不会捕获。
- 缓存内容：ACE handle、received indices/probs、最终 device `tokens_per_expert`、rank counts 及 shape/dtype 元数据。不缓存 hidden、permuted hidden、expert output 或 combine output。TE GroupedMLP 默认在 FC1 GroupGEMM 下发后于同一计算流 clone；其他 expert 路径回退独立 MUSA stream。
- 梯度：重计算 forward 用 `Buffer.dispatch(x, handle=...)` 返回新 hidden 和原 forward 的 received prob 数值；custom autograd backward 只执行一次 `Buffer.combine(..., topk_weights=grad_probs)`，把概率梯度送回本次重算的 router probs，不增加概率 A2A。
- 强制均衡：`RandomSTE` 改用 Megatron expert-parallel RNG tracker，使 checkpoint 原 forward/recompute 的随机路由完全一致；旧的独立 generator 不受 checkpoint RNG 恢复管理。
- 默认与回退：BF16、DeepEP-ACE、full recompute 或 selective MoE recompute 时默认启用；`DEEPEP_CACHE_RECOMPUTE_DISPATCH=0` 完整回退。FP8 暂不启用。结构 key 不一致 fail-fast；debug route mismatch 在所有 EP rank 统一走 full dispatch。
- 精度验证：单机 EP8、两层 MoE、seq4096、MBS2、BF16、full/uniform recompute、fake data、force-load-balancing。cache off/on 的 4-step aux loss 与总 grad norm 一致；第二层 router grad 完全一致，第一层 router grad norm 相对差约 `2.1e-6`。
- 性能验证：同配置、fused Router、NUMA 绑核的非 profiler 10-step（统计 step 3–10）均值为 cache-off `741.700 ms`、独立 cache stream `742.325 ms`、post-FC1 默认流 `741.100 ms`。post-FC1 相比独立流快 `1.225 ms/step`，相比 cache-off 快 `0.600 ms/step`；差异仍小于单次运行波动。8-rank trace 中两层 INT64 clone 累计均值从 `1.567 ms/rank` 降到 `0.056 ms/rank`，所有 clone 均位于 stream 0 且顺序为 `FC1 → clone → activation/FC2`。
- 显存：当前 shape 单 entry 为 `19,900,352` bytes（约 `18.98 MiB`），主要是 ACE row map 和 received indices/probs；总增量取决于每层同时在途的 checkpoint/microbatch 数，生产 PP16 上线前必须实测 warmup 峰值。

## 当前已知边界

- 单机 EP8 结果不能直接代表 ws128 的 TP2/PP8/EP64 生产拓扑；进入长训前仍需同配置短程 A/B。
- 当前 MATE main-grad 路径基于生产配置未开启 overlap-grad-reduce；若未来开启，需要重新验证梯度 ready 语义。
- DeepEP recompute cache 的单机收益目前与 metadata 快照开销基本相抵；其价值依赖生产 PP/EP 调度中的 CPU layout 空泡，不能按 microbenchmark 的 `0.47 ms/层` 直接外推。
- DeepEP recompute cache 当前仅覆盖 BF16 DeepEP-ACE；FP8/TE checkpoint 和异常跳过 backward 的长期生命周期仍需单独验证。
- CPU affinity 与 CPU NUMA/SNC/NPS 和容器 cpuset 强相关，必须按 `mate_cpu_affinity.md` 生成映射；错误映射会 fail-fast。
- MLA RoPE 精度目前验证了算子前后向和 4-step loss，尚未做长程收敛对比。
- 任何优化都必须同时观察空泡与 GEMM kernel 时间；只缩短 CPU 区间但拖慢 GroupGEMM 的方案不能保留。
