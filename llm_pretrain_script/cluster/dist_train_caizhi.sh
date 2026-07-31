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

# BF16 expert fast path（需要所有节点预装同版本 mate 与 mate-mubin）
# export MATE_GROUPED_GEMM=1       # fprop/dgrad=MATE，wgrad=TE grouped GEMM
# export MATE_USE_MAIN_GRAD=1      # wgrad 直写 FP32 main_grad，避免 BF16 临时梯度和 add
# export TE_TN_GM6_WGRAD=1         # 独立替换 TE BF16->FP32 grouped NT wgrad
# export MATE_TN_GM6_WGRAD=1       # MATE fprop/dgrad + GM6 wgrad；要求上面两个 MATE 开关为 1

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
