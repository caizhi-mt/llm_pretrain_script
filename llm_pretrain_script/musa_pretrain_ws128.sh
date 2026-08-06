#!/bin/bash
# 128 机 MUSA 预训练正式交付脚本（1024 卡，对齐 cuda_pretrain.sh）
#
# 并行: TP=2 + SP, PP=8, EP=64, GLOBAL_BATCH=NNODES×128（128 机 → 16384）, seq=4096
# 模型: 61 层 MoE+MLA；暂不能对齐的 cuda flag 见注释与 docs/musa_cuda_adaptation_issues.md
#
# 启动（auto_fault_manager --worldsize 128 已默认本入口，无需再 export ENTRY）:
#   LOG_NAME=ws128_YYYYMMDD bash auto_fault_manager.sh --hostfile ../hostfile.runtime.128 --worldsize 128 ...
#
# 双机缩小验证请用 musa_pretrain_ws2.sh，勿用本脚本。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hostfile}"

# ---------------------------------------------------------------------------
# hostfile → 集群拓扑（128 机：NNODES=128，由 hostfile / dist_run 注入）
# ---------------------------------------------------------------------------
if [ ! -f "${HOSTFILE}" ]; then
    echo "ERROR: hostfile 不存在: ${HOSTFILE}" >&2
    exit 1
fi

mapfile -t HOSTS < <(grep -v '^[[:space:]]*#' "${HOSTFILE}" | awk '{print $1}' | grep -v '^$')
NNODES=${NNODES:-${#HOSTS[@]}}
if [ "${NNODES}" -lt 1 ]; then
    echo "ERROR: hostfile 无有效节点: ${HOSTFILE}" >&2
    exit 1
fi

MASTER_ADDR=${MASTER_ADDR:-${HOSTS[0]}}
# 原 cuda_pretrain.sh: MASTER_PORT=22
# 现: 29500 — torchrun 分布式 rendezvous 端口，22 为 SSH 端口无法用于 NCCL/MCCL 建联
MASTER_PORT=${MASTER_PORT:-29500}

LOCAL_IP=$(ip -4 -o addr show bond0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
NODE_RANK=${RANK:-${NODE_RANK:-}}
if [ -z "${NODE_RANK}" ]; then
    rank=0
    for ip in "${HOSTS[@]}"; do
        if [ "${ip}" = "${LOCAL_IP}" ]; then
            NODE_RANK=${rank}
            break
        fi
        rank=$((rank + 1))
    done
fi
if [ -z "${NODE_RANK}" ] || [ "${NODE_RANK}" -ge "${NNODES}" ]; then
    echo "ERROR: 无法确定 NODE_RANK (LOCAL_IP=${LOCAL_IP}, hostfile=${HOSTFILE})" >&2
    exit 1
fi

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}   # 原: 8（不变）
export GPU_NUM=$((${GPUS_PER_NODE} * ${NNODES}))
export WORLD_SIZE=$((${GPUS_PER_NODE} * ${NNODES}))
export NODE_RANK MASTER_ADDR MASTER_PORT NNODES

# ---------------------------------------------------------------------------
# 环境变量（cuda_pretrain.sh → MUSA/MCCL 映射）
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export LD_LIBRARY_PATH=/usr/local/musa/lib:${LD_LIBRARY_PATH:-}
export MUSA_HOME=${MUSA_HOME:-/usr/local/musa}
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MUSA_KERNEL_TIMEOUT=${MUSA_KERNEL_TIMEOUT:-3200000}
export MUSA_BLOCK_SCHEDULE_MODE=${MUSA_BLOCK_SCHEDULE_MODE:-1}
export ACCELERATOR_BACKEND="musa"

# MCCL — JoyVideo /mnt/code/0407/XVideo/scripts/dist_train.sh 对齐（128 机千卡）
# bond0 socket 建联；不设 IB_HCA / CROSS_NIC / RoCE ens*
export MCCL_SOCKET_IFNAME=${MCCL_SOCKET_IFNAME:-bond0}
export MCCL_PROTOS=${MCCL_PROTOS:-2}
export MCCL_ALGOS=${MCCL_ALGOS:-1}
export MCCL_CHECK_POINTERS=${MCCL_CHECK_POINTERS:-0}
export MCCL_IB_GID_INDEX=${MCCL_IB_GID_INDEX:-3}
export MCCL_IB_TC=${MCCL_IB_TC:-122}
export MCCL_BUFFSIZE=${MCCL_BUFFSIZE:-20971520}   # 20MB：4MB 实测更差（PyTorch 侧 OOM 反增），8MB 仍不足
# collective channel 数钉为 8（+0.52% 实测）
#   注意库默认是 -2（拓扑自动），不是别处文档里说的常量 4；改前请用
#   MCCL_DEBUG=INFO MCCL_DEBUG_SUBSYS=INIT 抓真实生效值，默认值 != 生效值。
#   实测 16 与 8 完全持平，说明 8 已在拐点。
#   p2p channel 由 collective 推导：设 collective=8 后 p2p 自动变 8，
#   不要去设 MCCL_MIN_P2P_NCHANNELS —— 它会破坏数值正确性（iter1 grad norm
#   8845，正常 35.66），已验证并撤销。
#   为什么有效：这不是"重叠"，而是让 MCCL kernel 自身更快。本机 MCCL kernel
#   与计算实测零并发，kernel 耗时直接在关键路径上。前提是 overlap_grad_reduce
#   =False，梯度同步时 MCCL 独占 GPU，多占 SM 不挤压计算。
#   ⚠ 若将来开启梯度重叠，该前提消失，必须重测。
export MCCL_MIN_NCHANNELS=${MCCL_MIN_NCHANNELS:-8}
export MCCL_MAX_NCHANNELS=${MCCL_MAX_NCHANNELS:-8}
export MCCL_IB_TIMEOUT=${MCCL_IB_TIMEOUT:-19}
export MCCL_IB_RETRY_CNT=${MCCL_IB_RETRY_CNT:-7}
export MCCL_NET_SHARED_BUFFERS=${MCCL_NET_SHARED_BUFFERS:-0}
export MCCL_DEBUG=${MCCL_DEBUG:-WARN}

# 原 cuda: CUDA_DEVICE_MAX_CONNECTIONS=32
# 现: 1 — Megatron 在使用 TP/CP 时要求此值为 1（JoyVideo: MUSA_DEVICE_MAX_CONNECTIONS=1）
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export MUSA_DEVICE_MAX_CONNECTIONS=${MUSA_DEVICE_MAX_CONNECTIONS:-1}

export PYTORCH_MUSA_ALLOC_CONF=${PYTORCH_MUSA_ALLOC_CONF:-"expandable_segments:True"}  # 原: PYTORCH_CUDA_ALLOC_CONF
export TORCH_MCCL_AVOID_RECORD_STREAMS=${TORCH_MCCL_AVOID_RECORD_STREAMS:-1}
export TORCH_MCCL_TRACE_BUFFER_SIZE=${TORCH_MCCL_TRACE_BUFFER_SIZE:-1000000}  # 原: TORCH_NCCL_TRACE_BUFFER_SIZE

export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-8}   # 原: 8
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-8}   # 原: 8
export NVTE_DP_AMAX_REDUCE_INTERVAL=${NVTE_DP_AMAX_REDUCE_INTERVAL:-0}   # 原: 0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}  # 原: 1
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-0}                           # 原: 0
export NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN:-1}                           # 原: 1
export NVTE_EXT_MARGIN_SM=${NVTE_EXT_MARGIN_SM:-20}                    # 原: 20
export NVTE_DEBUG=${NVTE_DEBUG:-0}                                     # 原: 1；验证时降为 0 减少日志
export NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-0}                         # 原: 2
# 原 cuda: NVTE_NORM_*_USE_CUDNN / CUDNN_LOG* — MUSA 无 cuDNN，省略

export USE_MUSA_MOE=${USE_MUSA_MOE:-1}                                 # 原 cuda 无，musa_pretrain 新增
export USE_DEEPEP_ACE=${USE_DEEPEP_ACE:-1}                             # 对齐 telechat3/105B 参考脚本；=1 时 dispatcher 切 flex+deepep（见下方分支），=0 回退 alltoall
export USE_RECOMPUTE_VARIANCE=${USE_RECOMPUTE_VARIANCE:-0}
export ENABLE_D2H_IN_PERMUTATION=${ENABLE_D2H_IN_PERMUTATION:-0}
export NO_LOSS_REDUCE=${NO_LOSS_REDUCE:-1}                             # 原 cuda: 0；MUSA patch loss 上报格式不兼容标量写入

# ---------------------------------------------------------------------------
# 代码路径（对齐 cuda_pretrain.sh：固定 MCORE_PATH + pretrain_gpt.py，禁止 env 覆盖）
# cuda:  MCORE_PATH=/mnt/workspace/jdcloud/Megatron-LM
#        torchrun ... ${MCORE_PATH}/pretrain_gpt.py
# musa:  同路径语义固定为容器内 /home/Megatron-LM/pretrain_gpt.py；
#        仍经 LAUNCHER 先注入 musa_patch（MUSA 必需），再 runpy 到该入口。
# ---------------------------------------------------------------------------
MCORE_PATH=/home/Megatron-LM
PATCH_HOME=/home/megatron-lm-musa-patch
LAUNCHER=${SCRIPT_DIR}/pretrain_gpt_musa_launcher.py
PRETRAIN_SCRIPT=${MCORE_PATH}/pretrain_gpt.py
export MCORE_PATH
export PRETRAIN_SCRIPT
export PYTHONPATH=${MCORE_PATH}:${PATCH_HOME}:${PYTHONPATH:-}

if [ ! -d "${MCORE_PATH}/build" ]; then
    pushd "${MCORE_PATH}" >/dev/null
    python setup.py build_ext --inplace
    popd >/dev/null
fi

# ---------------------------------------------------------------------------
# 并行策略（512 卡实测最优：512 = TP1 × PP16 × CP1 × DP32，EP8）
#
# TP=1：MLA + MoE 下 TP 切分收益为负——TP>1 需要 all-reduce/all-gather，而本机
#       MCCL kernel 与计算 kernel 实测零并发，通信时间全额落在关键路径上。
# PP=16：61 层切 16 段，配合下方 decoder-first=3 / last=2 的非均匀切分。
# EP=8：DeepEP 在 EP>8 时跨节点 dispatch 必崩，EP≤8 走 intranode 才可用；
#       EP 增大也不省显存（MoE 静态权重占 77.7/80 GB 的结构性约束不变）。
# CP=1：seq4096 无需上下文并行。
# SP=0：TP=1 时 sequence-parallel 无意义（下方分支要求 TP>1 才会加该参数），
#       显式置 0 以免误开。
#
# A-006 已修 SP arity；前置：节点 megatron-lm-musa-patch/.../
# linear_with_grad_...py 的 SP return 须为 9 元组（TP>1 时才涉及）
# ---------------------------------------------------------------------------
TP=1
PP=16
CP=1
EP=8
MTP_LAYERS=0
MTP_LOSS=0.1
ENABLE_SEQUENCE_PARALLEL=${ENABLE_SEQUENCE_PARALLEL:-0}
# cuda LAYOUT Et|(tt|)*30L：128 全宽挂死（A-004），生产用 decoder-first/last 3+2

# ---------------------------------------------------------------------------
# 训练超参（与 cuda_pretrain.sh 对齐，可通过环境变量覆盖）
# ---------------------------------------------------------------------------
# MICRO_BATCH 直接线性缩放 1F1B 在途激活：stage1 峰值 ≈ 4 层 × 15 microbatch ×
# 单层单 mb 激活；MBS=2 时 261 GiB，MBS=1 时 131 GiB。关重计算前必须先降到 1。
MICRO_BATCH=${MICRO_BATCH:-2}
GLOBAL_BATCH=${GLOBAL_BATCH:-$((NNODES * 128))}
SEQ_LENGTH=${SEQ_LENGTH:-4096}                                         # 对齐 cuda / g2_128
# 重计算粒度：full = 每层整层重算（method/num-layers 生效）
#             selective = 只重算指定模块（method/num-layers 忽略）
# full 是被逼的，不是选出来的：block/N 与 selective 全部实测 OOM。selective 即使
# 挤进去也是余量≈0，而余量<1GB 会掉约 3% 吞吐，恰好抵消其收益。
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}                   # cuda 无；MUSA seq4096 必需
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}                          # 回退：block/N 显存不可行
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}                        # 回退：全部重算
DECAY_STEPS=100000
TRAINING_STEPS=${TRAINING_STEPS:-100000}
SAVE_INTERVAL=100000
LR_WARMUP_INIT=0.0
WARMUP_STEPS=1000
LR=2e-4
LR_MIN=2e-5
DECAY_STYLE=cosine
ADAM_BETA1=0.9
ADAM_BETA2=0.95
LB_RATE=1e-4
RB_RATE=1e-3
INIT_STD=0.006

# ---------------------------------------------------------------------------
# ENABLE_MOE_SHARED_EXPERT_OVERLAP=1 -> --moe-shared-expert-overlap   (+1.3%)
#   官方 5 段式 shared-expert / 通信重叠，已移植到 flex(DeepEP) dispatcher
#   （上游 transformer_config.py 的 assert 原本只允许 alltoall）。
#   shared expert 跑在自己的 stream 上：fc1 盖 dispatch a2a，fc2 盖 combine a2a，
#   并用 set_tensor_grad_fn_sequence_sr 抬高反向调度优先级使反向也重叠。
#
#   为什么有效：重叠对象是 DeepEP 的 memcpy（CE 路径），不是 MCCL kernel。
#   CE 与计算实测串行度 0.04（几乎完全并行）；而 MCCL kernel 与计算并发实测
#   0.00%，所以重叠类优化在本机只有走 CE 才可能有收益。
#
#   反向要单独补一刀：linear_fc1_forward_and_act(overlapped_comm_output) 的参数
#   不是数据输入，唯一用途是 set_tensor_grad_fn_sequence_sr(..., INT_MAX)。fc1
#   提前后无参可传，反向重叠会静默失效（约占一半收益）。解法是解耦——在
#   dispatch_postprocess 里单独对 dispatch 输出打优先级。
export ENABLE_MOE_SHARED_EXPERT_OVERLAP=${ENABLE_MOE_SHARED_EXPERT_OVERLAP:-1}
if [ "${ENABLE_MOE_SHARED_EXPERT_OVERLAP}" = "1" ]; then
    SHARED_EXPERT_OVERLAP_ARG="--moe-shared-expert-overlap"
    echo "[shared-expert-overlap] ENABLED (5 段式，fc1 盖 dispatch / fc2 盖 combine)"
else
    SHARED_EXPERT_OVERLAP_ARG=""
    echo "[shared-expert-overlap] disabled"
fi

# ---------------------------------------------------------------------------
# ENABLE_MOE_SE_LATE_ISSUE -> MOE_SE_LATE_ISSUE   0=关 1=只挪 fc1 2=fc1+fc2  (+1.08%)
#   把 shared expert 的下发从 a2a 调用【之前】挪到【之后】(fc1 进
#   dispatch_postprocess，fc2 进 combine_postprocess)。依赖上面那项。
#
#   前提修正:token_dispatch 只阻塞 CPU 到 ace_notify_dispatch 完成(要拿
#   num_recv_tokens 这个 host 数据),真正的 D2D payload 是在它【返回之后】才
#   异步跑的。rank32 iter6 实测:FusedDispatch 的 cpu_op 在 +18296us 结束,
#   payload 跑在 +18365..+20183us。所以 5 段式那句"在它之后下发什么也盖不住"
#   不成立 —— 恰恰相反,在它之前下发才盖不住。
#
#   改之前实测:fc1 中位比 payload 早 2.9ms 起跑、早 1.4ms 结束,fc2 中位比它
#   该盖的 combine 早 17.2ms。正向对 payload 覆盖率 0%(只盖住 20% 的元数据段),
#   反向 24%。不需要显式 event:CPU 阻塞本身就是同步器,流不可能执行 host 还
#   没下发的活。
#
#   实测:GBS=8192 下 175.15 -> 177.1(+1.08%,iter2-8 连续 7 点同向);
#   GBS=2048 下 133.65 -> 135.0(+1.01%)。loss iter1 偏离 5.0e-5,而同配置 run
#   自然离散 2.20e-4,比值 0.23。
#
#   level=2 实测与 level=1 持平略低,用 1:fc2 本来就跑在 expert grouped GEMM
#   底下被完全盖住、不花钱,挪它省不出没花的时间,还会把 fc2 排到 get_output()
#   前面。
export ENABLE_MOE_SE_LATE_ISSUE=${ENABLE_MOE_SE_LATE_ISSUE:-1}
if [ "${ENABLE_MOE_SE_LATE_ISSUE}" != "0" ]; then
    if [ "${ENABLE_MOE_SHARED_EXPERT_OVERLAP}" != "1" ]; then
        echo "Error: ENABLE_MOE_SE_LATE_ISSUE=${ENABLE_MOE_SE_LATE_ISSUE} 需要 ENABLE_MOE_SHARED_EXPERT_OVERLAP=1" >&2
        exit 1
    fi
    export MOE_SE_LATE_ISSUE=${ENABLE_MOE_SE_LATE_ISSUE}
    echo "[se-late-issue] ENABLED level=${MOE_SE_LATE_ISSUE} (1=只挪 fc1 / 2=fc1+fc2)"
else
    export MOE_SE_LATE_ISSUE=0
    echo "[se-late-issue] disabled"
fi

# ---------------------------------------------------------------------------
# ENABLE_MOE_SHARED_EXPERT_EARLY=1 -> MOE_SHARED_EXPERT_EARLY=1   【默认关】
#   方案C：在 MoELayer.custom_forward 里把 shared expert 提到 dispatch 之前算完，
#   再把结果透传给 experts_compute。实测约 +0.3%，已被上面的完整 5 段式取代，
#   二者【互斥】(完整版由 dispatcher 持有 shared_experts，此处会跳过)。
#   代码保留以防重复探索。
export ENABLE_MOE_SHARED_EXPERT_EARLY=${ENABLE_MOE_SHARED_EXPERT_EARLY:-0}
if [ "${ENABLE_MOE_SHARED_EXPERT_EARLY}" = "1" ]; then
    export MOE_SHARED_EXPERT_EARLY=1
    echo "[shared-expert-early] ENABLED (提前到 dispatch 之前下发)"
else
    export MOE_SHARED_EXPERT_EARLY=0
    echo "[shared-expert-early] disabled"
fi

# MOE_SE_DISPATCH_EVENT：给 a2a 传一个在 shared expert 入队【之前】记录的 event，
#   使 comm stream 不再等 fc1。实验39 实测 +0：那 211.2 ms 空隙里 GPU 有 99.7%
#   在跑 fc1，不是可回收的损失。代码保留，默认 0（token_dispatcher.py 读该变量）。
export MOE_SE_DISPATCH_EVENT=${MOE_SE_DISPATCH_EVENT:-0}

# ---------------------------------------------------------------------------
# 数据 / 输出路径（RUN_NAME 来自 LOG_NAME，避免 cache/ckpt 互相覆盖）
# ---------------------------------------------------------------------------
BASE=/mnt/code/llm_pretrain
RUN_NAME=${LOG_NAME:-"ws128-$(date +%Y%m%d_%H%M%S)"}
TOKENIZER_PATH=${TOKENIZER_PATH:-${BASE}/tokenizer}
SAVE_PATH=${SAVE_PATH:-/home/jd/wangkang/llm_pretrain/outputs}
LOG_OUTPUT=${LOG_OUTPUT:-/home/jd/wangkang/llm_pretrain/outputs/logs}
DATA_PATH=${DATA_PATH:-/home/jd/wangkang/llm_pretrain/data/tkn_ds_the_pile}

# ---------------------------------------------------------------------------
# 模型结构（MoE + MLA，对齐 cuda_pretrain.sh；MUSA 暂不能启用的 flag 见行内注释）
# ---------------------------------------------------------------------------
ADD_NETWORK_SIZE_ARGS=(
    --decoder-first-pipeline-num-layers 3
    --decoder-last-pipeline-num-layers 2
    --recompute-granularity ${RECOMPUTE_GRANULARITY}
    --recompute-method ${RECOMPUTE_METHOD}
    --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}
    --num-layers 61
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --kv-channels 128
    --position-embedding-type rope
    --rotary-base 10000
    --rotary-percent 1.0
    --rope-type rope
    --make-vocab-size-divisible-by 3232
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --multi-latent-attention
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --clip-grad 1.0
    --weight-decay 0.1
    --qk-layernorm
    --num-experts 256
    --manual-gc
    --manual-gc-interval 5
    --moe-layer-freq "([0]*3+[1]*58)"
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-router-pre-softmax
    --moe-router-group-topk 4
    --moe-router-num-groups 8
    --moe-router-topk-scaling-factor 2.5
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --init-method-std ${INIT_STD}
    --attention-backend flash
    --disable-bias-linear
    --moe-router-dtype fp32
    --transformer-impl transformer_engine
    --use-flash-attn
    --no-rope-fusion
    --cross-entropy-loss-fusion
    # 交叉熵融合实现走 TE 而非 native（对齐 musa_pretrain_worldsize512_caizhi.sh L587）：
    #   native → megatron 的 fused_vocab_parallel_cross_entropy
    #   te     → TE 的 parallel_cross_entropy（language_module.py L133 分支）
    # 词表 129280、seq 4096、MBS 2，logits 是全流程最大的中间张量之一，
    # TE 的实现把 max/sum/gather 压进一个 kernel，省掉 native 路径上的中间物化。
    # 依赖：TE 已装且 megatron.core.extensions.transformer_engine 能导出
    # te_parallel_cross_entropy（本集群 TE 2.0.0，已确认可导入）；导不到时
    # language_module.py L155 会直接抛 RuntimeError 而不是静默回退，所以换环境要留意。
    # 回退：把下面这行改回 native。
    --cross-entropy-fusion-impl te
    --moe-permute-fusion
    --moe-router-force-load-balancing
    ${SHARED_EXPERT_OVERLAP_ARG}
    # router 的 topk / sigmoid / group-limited-topk / aux-loss 打分融合成 TE kernel
    # （对齐 musa_pretrain_worldsize512_caizhi.sh L592）。
    #
    # 硬前提，两条缺一不可：
    #   1) 装的 TE 必须带 transformer_engine/pytorch/router.py（摩尔线程自编译
    #      wheel 有，镜像自带的 2.0.0+e8a0a52 没有）；
    #   2) pretrain_gpt_musa_launcher.py 里的 router-fusion 注入段必须生效——
    #      Megatron 用 is_te_min_version("2.7.0.dev") 卡这三个融合符号，而 musa
    #      TE 版本号是 2.0.0，过不了门，符号被置 None。
    # 两者任一不满足，router 前向会抛
    #   ValueError: fused_topk_with_score_function is not available.
    # 它不会静默回退到非融合实现。回退办法是注释掉下面这行（或 ROUTER_FUSION_BYPASS=0
    # 只关注入段——那样反而会踩上面的 ValueError，不要这么用）。
    --moe-router-fusion
)
# GroupGEMM（对齐 examples 各模型脚本的使能方式，即 --moe-grouped-gemm 参数）:
# MOE_GROUPED_GEMM=1（默认，原行为）→ 专家计算走 GroupedMLP
# MOE_GROUPED_GEMM=0 → 去掉该参数，回退 SequentialMLP（逐专家循环，仅排查问题用）
export MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-1}
if [ "${MOE_GROUPED_GEMM}" = "1" ]; then
    ADD_NETWORK_SIZE_ARGS+=(
        --moe-grouped-gemm
    )
fi
# MoE dispatcher（对齐 telechat3/105B run_pretrain_telechatv3_105B_musa.sh）:
# USE_DEEPEP_ACE=1 → flex + deepep + ACE（musa_patch deepep_ace，fused_a2a Buffer use_ace=True）
# USE_DEEPEP_ACE=0 → 回退原 alltoall 路径
if [ "${USE_DEEPEP_ACE}" = "1" ]; then
    ADD_NETWORK_SIZE_ARGS+=(
        --moe-token-dispatcher-type flex
        --moe-enable-deepep
        --moe-token-drop-policy probs
        --enable-experimental
    )
    # 参考脚本 flex+deepep 时 MCCL_CROSS_NIC=1（非 deepep 链路默认 0）
    export MCCL_CROSS_NIC=${MCCL_CROSS_NIC:-1}
else
    ADD_NETWORK_SIZE_ARGS+=(
        --moe-token-dispatcher-type alltoall
    )
fi
# 对齐 cuda：TP=2 时开 sequence-parallel（A-006）
if [ "${ENABLE_SEQUENCE_PARALLEL}" = "1" ] && [ "${TP}" -gt 1 ]; then
    ADD_NETWORK_SIZE_ARGS+=(
        --sequence-parallel
    )
fi
# 未启用（见 docs/musa_cuda_adaptation_issues.md）:
#   --pipeline-model-parallel-layout / overlap / delay-wgrad /
#   --moe-shared-expert-compute-before-router
# flex+deepep+ACE / --enable-experimental 已随 USE_DEEPEP_ACE=1 接入（见上方 dispatcher 分支）
# --moe-router-fusion 已启用（见上方 ADD_NETWORK_SIZE_ARGS，依赖 launcher 注入段）

if [ "${NNODES}" -lt 128 ]; then
    echo "WARNING: musa_pretrain_ws128.sh 预期 NNODES>=128，当前 NNODES=${NNODES}" >&2
fi

if [ "$MTP_LAYERS" -gt 0 ]; then
    ADD_NETWORK_SIZE_ARGS=(
        ${ADD_NETWORK_SIZE_ARGS[@]}
        --mtp-num-layers ${MTP_LAYERS}
        --mtp-loss-scaling-factor ${MTP_LOSS}
    )
fi

# ---------------------------------------------------------------------------
# DATA PROCESS（对齐 cuda_pretrain.sh 加权 blend 逻辑）
# 原 cuda: DATA_PATH=/mnt/workspace/data/merged，文件名如 *-merge.bin
# 现: 本地 DATA_PATH 为 tkn_ds_the_pile/*_text_document.bin，文件名不匹配 STAGE1_DATA
#     → 先走 cuda 同名匹配；若无匹配则 fallback 等权使用全部 *.bin
# ---------------------------------------------------------------------------
declare -A STAGE1_DATA=(
    ["2-3-merge.bin"]=86.683163068
    ["3-4-merge.bin"]=59.885228928
    ["CC-MAIN-2021-04-merge.bin"]=16.826519427
    ["CC-MAIN-2021-10-merge.bin"]=13.837581715
    ["CC-MAIN-2021-17-merge.bin"]=14.980002313
    ["CC-MAIN-2021-21-merge.bin"]=9.779054111
    ["CC-MAIN-2021-25-merge.bin"]=12.707104188
    ["CC-MAIN-2021-31-merge.bin"]=18.312108025
    ["CC-MAIN-2021-39-merge.bin"]=16.28938315
    ["CC-MAIN-2021-43-merge.bin"]=18.801128405
    ["CC-MAIN-2021-49-merge.bin"]=12.684432821
    ["CC-MAIN-2022-05-merge.bin"]=16.324092216
    ["CC-MAIN-2022-21-merge.bin"]=20.576187199
    ["CC-MAIN-2022-27-merge.bin"]=13.207847354
    ["CC-MAIN-2022-33-merge.bin"]=9.65958263
    ["CC-MAIN-2023-06-merge.bin"]=19.724194067
    ["CC-MAIN-2023-14-merge.bin"]=16.33230536
    ["CC-MAIN-2023-23-merge.bin"]=21.760590368
    ["CC-MAIN-2024-22-merge.bin"]=14.299276047
    ["CC-MAIN-2024-26-merge.bin"]=13.019718751
    ["CC-MAIN-2024-30-merge.bin"]=12.693607986
    ["CC-MAIN-2024-33-merge.bin"]=10.86458841
    ["CC-MAIN-2024-38-merge.bin"]=13.383161961
    ["CC-MAIN-2024-42-merge.bin"]=11.09886149
    ["CC-MAIN-2024-46-merge.bin"]=12.230566164
    ["CC-MAIN-2024-51-merge.bin"]=12.740120855
)
FILE_NAMES=()
while IFS= read -r file; do
    FILE_NAMES+=("${file}")
done < <(find "$DATA_PATH" -type f -name "*.bin" 2>/dev/null | sort)
DATA_PATHS=()
for file_name in "${FILE_NAMES[@]}"; do
    file_path=$file_name
    file_name_only="${file_path#"$DATA_PATH/"}"
    if [[ -v STAGE1_DATA["$file_name_only"] ]]; then
        weight=${STAGE1_DATA["$file_name_only"]}
    else
        continue
    fi
    length=${#file_path}
    DATA_PATHS+=("${weight} ${file_path:0:$((length - 4))}")
done
if [ ${#DATA_PATHS[@]} -eq 0 ]; then
    # fallback: 本地验证数据无 merge.bin 命名，等权使用可用分片
    # 原 cuda STAGE1_DATA 中所有 merge.bin 均有配套 idx；本地 02_text_document 缺 idx，需过滤
    echo "NOTE: STAGE1_DATA 无匹配文件，fallback 等权加载 ${DATA_PATH} 下 bin+idx 成对分片" >&2
    for file_name in "${FILE_NAMES[@]}"; do
        file_path=$file_name
        length=${#file_path}
        prefix="${file_path:0:$((length - 4))}"
        idx_path="${prefix}.idx"
        if [ -f "${idx_path}" ]; then
            DATA_PATHS+=("1.0 ${prefix}")
        fi
    done
fi
if [ ${#DATA_PATHS[@]} -eq 0 ]; then
    echo "ERROR: 未找到可用数据，请检查 DATA_PATH=${DATA_PATH}" >&2
    exit 1
fi
DATA_PATH_ARGUMENT=$(printf "%s " "${DATA_PATHS[@]}")
echo "DATA_PATH_ARGUMENT=${DATA_PATH_ARGUMENT}"

if [ ! -d "$SAVE_PATH" ]; then
  mkdir -p "$SAVE_PATH"
fi

SAVE_PATH=$SAVE_PATH/$RUN_NAME
LOG_OUTPUT=$LOG_OUTPUT/$RUN_NAME
mkdir -p $LOG_OUTPUT

# ---------------------------------------------------------------------------
# 分布式 / 训练 / 数据 / 日志参数（对齐 cuda_pretrain.sh）
# ---------------------------------------------------------------------------
DISTRIBUTED_ARGS=(
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size ${CP}
    --expert-tensor-parallel-size 1
    --expert-model-parallel-size ${EP}
    --use-distributed-optimizer
)

TRAINING_ARGS=(
    --use-mcore-models
    --micro-batch-size ${MICRO_BATCH}
    --global-batch-size ${GLOBAL_BATCH}
    --train-iters ${TRAINING_STEPS}
    --no-check-for-nan-in-loss-and-grad
    --max-position-embeddings ${SEQ_LENGTH}
    --lr-decay-iters ${DECAY_STEPS}
    --lr-warmup-iters ${WARMUP_STEPS}
    --lr-warmup-init ${LR_WARMUP_INIT}
    --lr ${LR}
    --min-lr ${LR_MIN}
    --lr-decay-style ${DECAY_STYLE}
    --adam-beta1 ${ADAM_BETA1}
    --adam-beta2 ${ADAM_BETA2}
    --moe-aux-loss-coeff ${LB_RATE}
    --moe-router-bias-update-rate ${RB_RATE}
    # 梯度累加融合已使能（对齐 musa_pretrain_worldsize512_caizhi.sh L1475：该行保持注释）。
    # 原先关掉是因为 MUSA patch 的 LinearWithGradAccumulationAndAsyncCommunication
    # 反向返回的梯度个数与前向入参对不上（expected 9 got 8）；现在 patch 里前向是
    # 9 个入参（input/weight/bias/gradient_accumulation_fusion/allreduce_dgrad/
    # sequence_parallel/grad_output_buffer/wgrad_deferral_limit/tp_group），反向两条
    # return 路径也都返回 9 个，这个不匹配已经不存在，不必再关。
    # 打开后 wgrad 由 fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32 直接累加进
    # weight.main_grad，省掉单独物化一个 BF16 grad_weight 再 add 到 main_grad 的往返；
    # 与 MATE_USE_MAIN_GRAD=1 是同一条思路，只是覆盖 patch 接管的那部分 linear。
    # 依赖：fused_weight_gradient_mlp_cuda 可导入（本集群已确认，导出
    # wgrad_gemm_accum_fp32 / wgrad_gemm_accum_fp16 两个符号）。
    # 生产 run 的参数 dump 里 gradient_accumulation_fusion = True，即以此配置在跑。
    #--no-gradient-accumulation-fusion                         # 需要回退时取消注释
    --eval-iters 0
    --eval-interval ${SAVE_INTERVAL}
    --save ${SAVE_PATH}
    --save-interval ${SAVE_INTERVAL}
    --init-method-std ${INIT_STD}
)

DATA_ARGS=(
    --seq-length ${SEQ_LENGTH}
    --data-cache-path ${SAVE_PATH}/cache
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_PATH}
    --data-path ${DATA_PATH_ARGUMENT}
    --split 100,0,0
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --num-workers 6                          # B2: 对齐 cuda（原 simu 为 2）
)

LOGGING_ARGS=(
    # 原 cuda: 以下 tensorboard 相关参数全开
    # 现: 关闭 tensorboard — musa_patch training_log 写入标量时维度报错 (size:2 vs 0-dim)
    --log-throughput
    --log-interval 1
    --logging-level 40
    --bf16
)
if [ "${ENABLE_TENSORBOARD:-0}" = "1" ]; then
    # 注意: --log-memory-to-tensorboard 在 MUSA 上会 KeyError('reserved_bytes.all.current')，勿开
    LOGGING_ARGS+=(
        --log-timers-to-tensorboard
        --log-num-zeros-in-grad
        --log-params-norm
        --log-validation-ppl-to-tensorboard
        --tensorboard-dir ${SAVE_PATH}/tensorboard
        --tensorboard-log-interval 1
        --moe-per-layer-logging
    )
fi

# ---------------------------------------------------------------------------
# Profiler（参考 telechat3/105B run_pretrain_telechatv3_105B_musa.sh）
# ENABLE_PROFILER=1 时：导出 MUSA profiler 环境变量 + Megatron --profile 区间
# 注意：profile 区间内每 step 都会 dump trace，正式训练勿长开
# ---------------------------------------------------------------------------
PROFILE_ARGS=()
if [ "${ENABLE_PROFILER:-0}" = "1" ]; then
    export ENABLE_PROFILER
    export PROFILER_FREQ=${PROFILER_FREQ:-4}
    export PROFILER_WARMUP_STEPS=${PROFILER_WARMUP_STEPS:-3}
    export PROFILER_PROFILE_MEMORY=${PROFILER_PROFILE_MEMORY:-1}
    # MUSA_LAUNCH_BLOCKING 由入口显式 export 才生效（显著拖慢，默认不开）
    PROFILE_ARGS+=(
        --profile
        --profile-step-start ${PROFILE_STEP_START:-4}
        --profile-step-end ${PROFILE_STEP_END:-6}
    )
fi

FILE=${SAVE_PATH}/latest_checkpointed_iteration.txt
if [ -f "$FILE" ]; then
    INPUT=(--load ${SAVE_PATH})
else
    INPUT=()
fi

echo "========================================"
echo "MUSA ws128 正式交付训练 (rank ${NODE_RANK}/${NNODES})"
echo "  LOCAL_IP   : ${LOCAL_IP}"
echo "  MASTER     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  WORLD_SIZE : ${WORLD_SIZE}"
echo "  ENTRY      : ${PRETRAIN_SCRIPT}  (via ${LAUNCHER})"
echo "  TP/PP/EP   : ${TP}/${PP}/${EP}"
echo "  SEQ_PARALLEL: ${ENABLE_SEQUENCE_PARALLEL}"
echo "  SEQ_LENGTH : ${SEQ_LENGTH}"
echo "  GLOBAL_BS  : ${GLOBAL_BATCH}"
echo "  TRAIN_ITERS: ${TRAINING_STEPS}"
echo "  PROFILER   : ${ENABLE_PROFILER:-0}"
echo "  DEEPEP_ACE : ${USE_DEEPEP_ACE}"
echo "  GROUP_GEMM : ${MOE_GROUPED_GEMM}"
echo "  RUN_NAME   : ${RUN_NAME}"
echo "  LOG        : ${LOG_OUTPUT}/output_rank${NODE_RANK}.log"
echo "========================================"

# 原 cuda: nohup torchrun ... pretrain_gpt.py
# 现: 经 LAUNCHER 注入 MUSA patch；FOREGROUND=1 可前台调试
if [ "${FOREGROUND:-0}" = "1" ]; then
    torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ${LAUNCHER} \
        ${DISTRIBUTED_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${ADD_NETWORK_SIZE_ARGS[@]} \
        ${LOGGING_ARGS[@]} \
        ${PROFILE_ARGS[@]} \
        ${INPUT[@]} 2>&1 | tee $LOG_OUTPUT/output_rank${NODE_RANK}.log
else
    nohup torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ${LAUNCHER} \
        ${DISTRIBUTED_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${ADD_NETWORK_SIZE_ARGS[@]} \
        ${LOGGING_ARGS[@]} \
        ${PROFILE_ARGS[@]} \
        ${INPUT[@]} > $LOG_OUTPUT/output_rank${NODE_RANK}.log 2>&1 &
    echo "已后台启动 torchrun, PID=$!, 日志: $LOG_OUTPUT/output_rank${NODE_RANK}.log"
fi
