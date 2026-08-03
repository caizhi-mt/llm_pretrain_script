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
# export USE_DEEPEP_ACE=0          # 置 0 回退 alltoall dispatcher（关闭 flex+deepep+ACE）

# GroupGEMM（对齐 examples 各模型脚本 --moe-grouped-gemm；musa_pretrain_ws128.sh 默认已开）
# export MOE_GROUPED_GEMM=0        # 置 0 去掉 --moe-grouped-gemm，回退 SequentialMLP（仅排查问题用）

# BF16 expert fast path（默认开启；需要所有节点预装同版本 mate 与 mate-mubin）
# export MATE_GROUPED_GEMM=0       # 置 0 回退 Transformer Engine GroupedLinear
# export MATE_USE_MAIN_GRAD=0      # 置 0 禁用 wgrad 直写 FP32 main_grad
# export MATE_FLASH_ATTN=0         # 置 0 回退原生 MUSA FlashAttention 前向
# export MATE_CACHE_MUBIN_DISPATCH=0  # 置 0 禁用 GroupGEMM/FA MUBIN 元数据缓存
# export MATE_DEFER_DEEPEP_COUNTS=0   # 置 0 回退 DeepEP counts 同步构造路径
# export MUSA_NATIVE_ROPE=0            # 置 0 回退标准 RoPE eager 组合算子
# export MUSA_FUSED_MLA_ROPE=0         # 置 0 仅使用 torch.rope，不融合 MLA Q/K/V 布局

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
