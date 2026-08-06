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
