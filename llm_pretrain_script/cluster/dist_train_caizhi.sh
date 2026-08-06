export LOG_NAME=ws128_$(date +%Y%m%d)   # 每次新 run 换名，避免 cache/ckpt 冲突

# Profiler（参考 telechat3/105B run_pretrain_telechatv3_105B_musa.sh；需要抓 profile 时取消注释）
# export ENABLE_PROFILER=1          # 总开关：同时启用 MUSA profiler 环境变量 + Megatron --profile
# export PROFILER_FREQ=4
# export PROFILER_WARMUP_STEPS=3
# export PROFILER_PROFILE_MEMORY=1
# export MUSA_LAUNCH_BLOCKING=1    # 显著拖慢训练，仅精确定位 kernel 时再开
# export PROFILE_STEP_START=4      # Megatron --profile-step-start（默认 4）
# export PROFILE_STEP_END=6        # Megatron --profile-step-end（默认 6）

# DeepEP-ACE（参考 telechat3/105B 脚本 L49；musa_pretrain_ws128.sh 默认已开）
#   在入口显式置 1，不再依赖 musa_pretrain_ws128.sh 的默认值：
#   dist_run_megatron.sh 以 USE_DEEPEP_ACE=${USE_DEEPEP_ACE:-} 透传，空串会让每个节点
#   各自回落到脚本默认，开关的真实取值在入口脚本里看不出来。显式 export 后
#   dist_run 透传的是确定值，日志里 "DEEPEP_ACE : 1" 也才有对照意义。
#   =1 时 dispatcher 走 flex + deepep + ACE（musa_patch deepep_ace；
#   fused_a2a.get_buffer 据此构造 Buffer(use_ace=True, num_ace_buffers=1, train_mode=True)）。
#
#   注意：置 0 的 alltoall 回退路径在本配置（TP1/PP16/EP8/CP1，world=512）下**起不来**。
#   实验42 两次均死在 MCCL communicator 初始化 / McclUniqueId，与残留进程无关。
#   所以它目前只是名义上的回退开关，不能当作可用的对照组——需要 alltoall 对照时
#   得先解决 MCCL 初始化问题。
export USE_DEEPEP_ACE=1          # 置 0 回退 alltoall dispatcher（当前该路径不可用）

# MLA q/kv 下投影融合（实验35，移植 PR#2 c17e7f0；musa_pretrain_ws128.sh 默认 0）
#   +0.12%，iter1 gnorm=35.668 已复核。置 0 回退原双 linear 路径。
export MUSA_FUSED_MLA_DOWN_PROJ=1

# DeepEP compact permute（实验36，移植 PR#2 96da2cb；musa_pretrain_ws128.sh 默认 0）
#   permute/unpermute kernel 只扫 topk=8 列而非 32 个 local-expert 列。
#   注意 triton 不可用时是静默回退，换环境要确认 import triton 成功。
export MUSA_COMPACT_PERMUTE=1

# MATE 提交线程绑核（本机实测拓扑, 不是文档里的 Intel 示例）
#   AMD EPYC 9T34, 2 socket x 64 core x 2 SMT = 256 逻辑 CPU, NPS1 -> 2 个 NUMA node
#   node0 = 0-63,128-191   node1 = 64-127,192-238,240-255   (CPU 239 offline)
#   物理核首线程: node0 -> 0-63, node1 -> 64-127 (SMT sibling = +128, 起步不选)
#   mthreads-gmi topo -m: GPU0-3 -> NUMA 0, GPU4-7 -> NUMA 1
#   每 rank 取 8 个连续物理核 = 恰好 1 个 CCD/L3 域(已用 lscpu -e CACHE 验证)
#   注意 mate 模式依赖 MATE_GROUPED_GEMM=1（绑定发生在 MATE 前向里）。
export MUSA_CPU_AFFINITY=1
export MUSA_CPU_AFFINITY_MODE=mate
export MUSA_CPU_AFFINITY_MAP='0-7;8-15;16-23;24-31;64-71;72-79;80-87;88-95'

# BF16 expert fast path（需要所有节点预装同版本 mate 与 mate-mubin）
export MATE_GROUPED_GEMM=1       # fprop/dgrad=MATE, wgrad=TE grouped GEMM
export MATE_USE_MAIN_GRAD=1      # wgrad 直写 FP32 main_grad, 避免 BF16 临时梯度和 add
# export MATE_DEFER_DEEPEP_COUNTS=0   # 置 0 回退 DeepEP counts 同步构造路径

# MATE MLA FlashAttention 前向（实验37，移植 PR#2 a82e08c；musa_pretrain_ws128.sh 默认 0）
#   +0.46%，但 iter1 grad norm 35.852 超出 35.66±0.03 判据 —— 这是本清单里
#   唯一有真实精度代价的一项，用户确认"精度可放宽"后才开。不接受就置 0。
export MATE_FLASH_ATTN=1
# export MATE_CACHE_MUBIN_DISPATCH=0  # 置 0 禁用 MUBIN 元数据缓存

# GroupGEMM（对齐 examples 各模型脚本 --moe-grouped-gemm；musa_pretrain_ws128.sh 默认已开）
# export MOE_GROUPED_GEMM=0        # 置 0 去掉 --moe-grouped-gemm，回退 SequentialMLP（仅排查问题用）

bash auto_fault_manager.sh \
  --hostfile ../hostfile.runtime.128 \
  --worldsize 128 \
  --dist-run ./dist_run_megatron.sh \
  --output-dir /home/jd/wangkang/llm_pretrain/outputs/${LOG_NAME} \
  --startup-grace 1800 \
  --hang-minutes 60 \
  --log-error-patterns "RuntimeError,ConnectionError,Segmentation fault,Out of memory,Traceback" \
  --skip-initial-netcheck \
  --daemon
