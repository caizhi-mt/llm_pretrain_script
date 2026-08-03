# llm_pretrain_script

JD 集群 128 机(1024 卡)MUSA LLM 预训练启动脚本链,取自 pod 内 `/mnt/code/llm_pretrain` 仓库(commit `b4c3470`)。

运行环境:namespace `his-test`,Deployment `jd-llm-pretrain-test`,pod 内工作目录 `/mnt/code/llm_pretrain`。

## 调用链

```
cluster/dist_train_caizhi.sh                       ← 训练入口(daemon 方式拉起)
└─ cluster/auto_fault_manager.sh                   ← 容错守护:起停、hang 检测、故障节点剔除重启
   ├─ cluster/dist_run_megatron.sh                 ← 按 hostfile 逐节点 SSH 分发
   │  └─ scripts/dist_train_megatron_ws128.sh      ← 每节点入口(--worldsize 128 时默认注入)
   │     └─ musa_pretrain_ws128.sh                 ← 实际训练脚本,组装参数后 torchrun
   │        ├─ pretrain_gpt_musa_launcher.py       ← launcher:先 import musa_patch 再 runpy 训练入口
   │        ├─ tokenizer/                          ← HuggingFaceTokenizer(DeepSeek 系)
   │        ├─ /home/Megatron-LM/pretrain_gpt.py   ← 容器内代码(本仓库 Megatron-LM/ 为其拷贝)
   │        └─ /home/megatron-lm-musa-patch/       ← 容器内代码(本仓库 megatron-lm-musa-patch/ 为其拷贝)
   ├─ cluster/hang_detect.sh                       ← 训练 hang 检测(内部调 mccl_bench/stop_all)
   ├─ cluster/mccl_bench.sh                        ← MCCL 通信探测
   └─ cluster/stop_all.sh                          ← 全节点停止
cluster/stop_train_caizhi.sh                       ← 停止入口(fault_manager --stop + stop_all)
hostfile.runtime.128                               ← 128 节点列表
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `cluster/dist_train_caizhi.sh` | 启动入口:`LOG_NAME=ws128_日期`,经 auto_fault_manager 以 `--daemon` 拉起,startup-grace 1800s,hang 检测 60min |
| `cluster/stop_train_caizhi.sh` | 停止入口:fault_manager `--stop`、杀 pid、`stop_all.sh` |
| `cluster/auto_fault_manager.sh` | 容错管理主体(~2100 行):监控日志错误模式、hang 检测、坏节点写 `error_node.txt` 后换节点重启 |
| `cluster/dist_run_megatron.sh` | 分发器:解析 hostfile,选主节点与空闲端口,逐节点 `ssh` 后台执行每节点入口 |
| `cluster/hang_detect.sh` / `cluster/mccl_bench.sh` / `cluster/stop_all.sh` | fault_manager 的探测与停止工具 |
| `scripts/dist_train_megatron_ws128.sh` | 每节点入口(128 机):导出 MUSA/MCCL 环境变量、ulimit,exec `musa_pretrain_ws128.sh` |
| `scripts/dist_train_megatron.sh` | 每节点入口(通用/双机验证,默认 `musa_pretrain_ws2.sh`) |
| `musa_pretrain_ws128.sh` | 128 机正式训练:TP=2+SP、PP=8、EP=64,GBS=16384,seq=4096,61 层 MoE+MLA,对齐 `cuda_pretrain.sh` |
| `musa_pretrain_ws2.sh` | 双机缩小验证版 |
| `pretrain_gpt_musa_launcher.py` | 训练 launcher:强制先 `import musa_patch`,再 runpy `${MCORE_PATH}/pretrain_gpt.py` |
| `hostfile.runtime.128` | 运行时 128 节点 IP 列表 |
| `tokenizer/` | tokenizer 模型与配置(`--tokenizer-type HuggingFaceTokenizer`) |

## 使用方法(pod 内)

```bash
cd /mnt/code/llm_pretrain/cluster

# 启动(每次新 run 会以 ws128_日期 命名,避免 cache/ckpt 冲突)
bash dist_train_caizhi.sh

# 停止
bash stop_train_caizhi.sh
```

## Profiler

参考 `telechat_train/megatron-lm-musa-patch/examples/telechat3/105B/run_pretrain_telechatv3_105B_musa.sh` 的做法,支持一键开启 profiler。默认全部关闭,行为与未改动前一致。

开启方式:取消 `cluster/dist_train_caizhi.sh` 顶部对应注释后正常启动:

```bash
export ENABLE_PROFILER=1          # 总开关:导出 MUSA profiler 环境变量 + Megatron --profile
export PROFILER_FREQ=4            # 可选,默认 4
export PROFILER_WARMUP_STEPS=3    # 可选,默认 3
export PROFILER_PROFILE_MEMORY=1  # 可选,默认 1
export MUSA_LAUNCH_BLOCKING=1     # 可选,显著拖慢训练,仅精确定位 kernel 时开
export PROFILE_STEP_START=4       # 可选,Megatron --profile-step-start,默认 4
export PROFILE_STEP_END=6         # 可选,Megatron --profile-step-end,默认 6
```

实现要点:

- `cluster/dist_train_caizhi.sh`:入口处集中放置上述开关(默认注释)。
- `cluster/dist_run_megatron.sh`:SSH 分发只透传白名单环境变量,已把 7 个 profiler 变量加入捕获与透传列表。
- `musa_pretrain_ws128.sh`:`ENABLE_PROFILER=1` 时导出 profiler 环境变量,并向 torchrun 追加 `--profile --profile-step-start/--profile-step-end`;启动横幅打印 `PROFILER: 0/1`。
- 每节点入口脚本无需改动,环境变量随 `exec` 自然传递。
- 注意:profile 区间内每 step 都会 dump trace,正式长训勿长开;`musa_pretrain_ws2.sh` 双机验证版暂未接入。

## DeepEP-ACE

参考 `telechat3/105B/run_pretrain_telechatv3_105B_musa.sh`(L49 `export USE_DEEPEP_ACE=1`)接入 DeepEP-ACE 优化,**默认开启**。

`USE_DEEPEP_ACE` 环境变量只在 flex dispatcher + DeepEP 的 `fused_a2a` 路径生效(musa_patch 按需加载 `deepep_ace` 模块,DeepEP Buffer 以 `use_ace=True` 创建),因此接入时同步做了 dispatcher 切换,`musa_pretrain_ws128.sh` 中按开关分支:

- `USE_DEEPEP_ACE=1`(默认):`--moe-token-dispatcher-type flex --moe-enable-deepep --moe-token-drop-policy probs --enable-experimental`,并默认 `MCCL_CROSS_NIC=1`(对齐参考脚本 flex+deepep 链路)。
- `USE_DEEPEP_ACE=0`:回退原 `--moe-token-dispatcher-type alltoall` 路径(改动前行为)。回退开关在 `cluster/dist_train_caizhi.sh` 顶部(默认注释),经 `dist_run_megatron.sh` SSH 白名单透传。

注意:参考脚本还开了 `--moe-router-fusion`,本仓库暂未接入(见 `docs/musa_cuda_adaptation_issues.md` 未启用清单);首次切 flex+deepep 建议先小步数验证再进长训。

## GroupGEMM

`megatron-lm-musa-patch/examples` 各模型脚本使能 group_gemm 的方式即 Megatron 参数 `--moe-grouped-gemm`(patch 侧无需额外模块)。本仓库 `musa_pretrain_ws128.sh` 原本写死开启,现改为环境变量 `MOE_GROUPED_GEMM` 控制:

- `MOE_GROUPED_GEMM=1`(默认,与原行为一致):追加 `--moe-grouped-gemm`,专家计算走 GroupedMLP。
- `MOE_GROUPED_GEMM=0`:去掉该参数,回退 SequentialMLP(逐专家循环,性能差,仅排查 grouped gemm 相关问题时用)。

回退开关在 `cluster/dist_train_caizhi.sh` 顶部(默认注释),经 `dist_run_megatron.sh` SSH 白名单透传;启动横幅打印 `GROUP_GEMM: 0/1`。

## MATE expert BF16 fast path

在 BF16、无 bias 的 GroupedMLP 上提供可回退的混合实现:

- fprop/dgrad: MATE `ragged_m_moe_gemm_16bit`;
- wgrad:单次 Transformer Engine `general_grouped_gemm(layout="NT")`;
- wgrad 直接写 FP32 `main_grad`,不创建 BF16 临时梯度,也不增加后续 BF16→FP32 add/cast。

该路径只局部接管 MoE expert GroupedLinear。ws128 现有的全局
`--no-gradient-accumulation-fusion` 保持不变,因此不会改变 dense Linear 的反向路径。

启用方式:

```bash
export MATE_GROUPED_GEMM=1
export MATE_USE_MAIN_GRAD=1
export MATE_FLASH_ATTN=1
```

两个变量经 `cluster/dist_run_megatron.sh` 的 SSH 白名单传到所有节点。设置
`MATE_GROUPED_GEMM=0` 可完整回退原 Transformer Engine GroupedLinear。

依赖与限制:

- 每个节点必须安装同版本的 `mate` 与 `mate-mubin`;启动脚本会检查并打印版本。
- 当前仅支持 BF16、连续 MUSA tensor、无 bias、非 FP8、DeepEP/GroupedMLP 路径;不满足条件时会打印一次 fallback 并走原 TE 实现。
- 当前生产配置未开启 overlap-grad-reduce。后续若开启该功能,需要先补充 direct-main-grad 的梯度 ready 验证。
- MATE 使用 `backend="mubin"`;不要只安装 `mate` 后让 128 节点在首次 kernel 时并发下载产物。

`MATE_CACHE_MUBIN_DISPATCH=1` 默认缓存不可变的 MUBIN module/dispatcher、已校验
artifact，以及按最终 `GemmMubinId` 选择的 kernel path。每一步仍使用当前动态
`M` 和 routing counts 选择 ASM id，并把当前 input/weight/output/counts 直接传给
launch；cache 不持有 tensor、data pointer 或专家 token 分布。设为 `0` 可回退
MATE 原生逐次 dispatch，仅用于 A/B 或故障排查。

## MATE MLA FlashAttention forward

DeepSeek MLA 的 BF16 fixed-length attention 默认使用混合实现：

- forward：MATE 0.2.5 MUBIN FlashAttention；
- backward：保留原生 MUSA `aten::_scaled_dot_product_attention_flash_musa_backward`；
- dispatch cache：复用 `MATE_CACHE_MUBIN_DISPATCH=1`，缓存不可变的 MUBIN
  artifact 选择和 launch handle，不缓存 Q/K/V、LSE 或 attention 输出。

快路径只接管 `Dqk=192/Dv=128`、causal、dropout=0、无 ALiBi/softcap、
CP=1、`USE_RECOMPUTE_VARIANCE=0` 的 MUSA BF16 BSHD 输入；其他配置完整回退
原 Transformer Engine FlashAttention。当前私有 MUBIN API 固定验证
`mate=mate-mubin=0.2.5`，版本不匹配时保持原生路径。

```bash
# 默认开启；完整回退原生 MUSA FlashAttention
export MATE_FLASH_ATTN=0

# 同时关闭 GroupGEMM/FA 的 MUBIN dispatch cache，仅用于 A/B
export MATE_CACHE_MUBIN_DISPATCH=0
```

该实现会保存 MATE forward 的 output/LSE，并在 backward 以 BHSD view 交给原生
MUSA kernel；不会调用 MATE 0.2.5 自带的 varlen backward。

## MUSA MLA RoPE fast path

标准 `--rope-type rope`、MLA、BF16 训练默认开启两级 MUSA 优化:

- `MUSA_NATIVE_ROPE=1`:未使用 MLA 布局融合时,Q/K RoPE 走 MUDNN `torch.rope`,替代 eager `cos/sin/mul/cat` 组合算子。
- `MUSA_FUSED_MLA_ROPE=1`:一次完成 Q RoPE 以及 KV split、K RoPE/broadcast 和 Q/K/V 连续布局,去掉 attention 前的 Q/K `cat` 与 V `contiguous`。MUSA dQ 使用 16-head tile;CUDA 保持原 tile。

当前融合布局仅对标准 RoPE、MUSA BF16/FP16、CP=1、非 packed sequence、非 inference 生效,且 QK/位置/V head dim 必须为 2 的幂;其他配置安全回退原路径。两个变量均由 `cluster/dist_run_megatron.sh` 透传,并由 `musa_pretrain_ws128.sh` 校验为 `0/1`。

回退方式:

```bash
# 只关闭 MLA Q/K/V 布局融合,保留 MUDNN torch.rope
export MUSA_FUSED_MLA_ROPE=0

# 完整回退 eager 标准 RoPE;同时不会启用 MLA 布局融合
export MUSA_NATIVE_ROPE=0
```

单机 8×S5000、EP8、1 层 MoE、seq=4096、MBS=2、BF16、fake data 的 8-rank trace 中,Profiler step 中位数由 541.895 ms 降至 538.808 ms(-0.57%),GPU active union 均值减少 3.46 ms。4-step loss 最大绝对差为 `1.3e-4`,无 NaN/skipped iteration。该数据只用于验证单机算子和短程 loss;128 机生产拓扑仍需独立 A/B。

关键路径(pod 内):

- 训练输出/ckpt:`/home/jd/wangkang/llm_pretrain/outputs/${LOG_NAME}`
- 数据:`/home/jd/wangkang/llm_pretrain/data/tkn_ds_the_pile`
- Megatron 代码:`/home/Megatron-LM`(首次运行自动 `setup.py build_ext --inplace`)
- MUSA patch:`/home/megatron-lm-musa-patch`
