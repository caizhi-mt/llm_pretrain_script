#!/bin/bash
# Megatron llm_pretrain dist runner — mirrors dist_run_fsdp8.sh (JoyVideo 千卡模式).
# Per-node SSH → scripts/dist_train_megatron.sh (NOT launch_multi2.sh per node).
#
# Usage:
#   bash dist_run_megatron.sh HOSTFILE [--logdir LOG_DIR] [--output-dir OUTPUT_DIR]

set -euo pipefail

TRAINING_STEPS="${TRAINING_STEPS-}"
LOG_NAME="${LOG_NAME-}"
ENABLE_TENSORBOARD="${ENABLE_TENSORBOARD-}"
MUSA_PRETRAIN_ENTRY="${MUSA_PRETRAIN_ENTRY-}"
ENABLE_SEQUENCE_PARALLEL="${ENABLE_SEQUENCE_PARALLEL-}"
# Profiler 透传（参考 telechat3/105B；由 dist_train_caizhi.sh 等入口 export）
ENABLE_PROFILER="${ENABLE_PROFILER-}"
PROFILER_FREQ="${PROFILER_FREQ-}"
PROFILER_WARMUP_STEPS="${PROFILER_WARMUP_STEPS-}"
PROFILER_PROFILE_MEMORY="${PROFILER_PROFILE_MEMORY-}"
MUSA_LAUNCH_BLOCKING="${MUSA_LAUNCH_BLOCKING-}"
PROFILE_STEP_START="${PROFILE_STEP_START-}"
PROFILE_STEP_END="${PROFILE_STEP_END-}"
# DeepEP-ACE 开关透传（musa_pretrain_ws128.sh 默认 1；置 0 回退 alltoall）
USE_DEEPEP_ACE="${USE_DEEPEP_ACE-}"
# GroupGEMM 开关透传（musa_pretrain_ws128.sh 默认 1；置 0 回退 SequentialMLP）
MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM-}"
# shared-expert / 通信重叠开关透传（默认值都在 musa_pretrain_ws128.sh 里）
#   注意必须透传【顶层】开关 ENABLE_MOE_SE_LATE_ISSUE，不能透传派生的
#   MOE_SE_LATE_ISSUE —— musa_pretrain_ws128.sh 在每个节点上最后执行，会把派生值
#   覆盖掉，形成静默失效。已踩过一次。
# MUDNN torch.rope 快路径开关透传（musa_patch/rotary_pos_embedding.py 内默认 1）
#   不透传的话各节点只能用代码里的默认值，想临时关掉做对照就没有入口。
MUSA_NATIVE_ROPE="${MUSA_NATIVE_ROPE-}"
# MLA q/kv 下投影融合开关透传（musa_pretrain_ws128.sh 默认 0，由入口脚本 export 1）
MUSA_FUSED_MLA_DOWN_PROJ="${MUSA_FUSED_MLA_DOWN_PROJ:-0}"
# DeepEP compact permute 开关透传（musa_pretrain_ws128.sh 默认 0，由入口脚本 export 1）
MUSA_COMPACT_PERMUTE="${MUSA_COMPACT_PERMUTE:-0}"
# MATE MLA FlashAttention 前向开关透传。
#   必须给默认值：musa_patch 里 env_flag 的默认是 "1"，且对非 0/1 值直接 raise，
#   传空串会让每个节点在 import musa_patch 时炸 ValueError。
# 绑核开关透传。MAP 里含 ';'，ssh 那行必须把它整体加引号，否则远端 shell 会
# 把分号当命令分隔符，后面的赋值和 bash 调用全部被截断。
MUSA_CPU_AFFINITY="${MUSA_CPU_AFFINITY:-0}"
MUSA_CPU_AFFINITY_MODE="${MUSA_CPU_AFFINITY_MODE:-mate}"
MUSA_CPU_AFFINITY_MAP="${MUSA_CPU_AFFINITY_MAP-}"
MATE_GROUPED_GEMM="${MATE_GROUPED_GEMM:-0}"
MATE_USE_MAIN_GRAD="${MATE_USE_MAIN_GRAD:-1}"
MATE_DEFER_DEEPEP_COUNTS="${MATE_DEFER_DEEPEP_COUNTS:-1}"
MATE_FLASH_ATTN="${MATE_FLASH_ATTN:-0}"
MATE_CACHE_MUBIN_DISPATCH="${MATE_CACHE_MUBIN_DISPATCH:-1}"
ENABLE_MOE_SHARED_EXPERT_OVERLAP="${ENABLE_MOE_SHARED_EXPERT_OVERLAP-}"
ENABLE_MOE_SE_LATE_ISSUE="${ENABLE_MOE_SE_LATE_ISSUE-}"
ENABLE_MOE_SHARED_EXPERT_EARLY="${ENABLE_MOE_SHARED_EXPERT_EARLY-}"
MOE_SE_DISPATCH_EVENT="${MOE_SE_DISPATCH_EVENT-}"

HOSTFILE=""
LOG_DIR=""
OUTPUT_DIR=""
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-}"
COUNT=0

WORK_DIR="${LLM_PRETRAIN_WORK_DIR:-/mnt/code/llm_pretrain}"
DIST_TRAIN="${LLM_PRETRAIN_DIST_TRAIN:-scripts/dist_train_megatron.sh}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --logdir) LOG_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 HOSTFILE [--logdir LOG_DIR] [--output-dir OUTPUT_DIR]"
      exit 0
      ;;
    *)
      if [[ -z "$HOSTFILE" ]]; then
        HOSTFILE="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$HOSTFILE" ]]; then
  echo "Error: HOSTFILE is required" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ "$HOSTFILE" != /* ]]; then
  if [[ -f "$SCRIPT_DIR/$HOSTFILE" ]]; then
    HOSTFILE=$(cd "$SCRIPT_DIR" && realpath "$HOSTFILE")
  elif [[ -f "$WORK_DIR/$HOSTFILE" ]]; then
    HOSTFILE=$(cd "$WORK_DIR" && realpath "$HOSTFILE")
  fi
fi

if [[ ! -f "$HOSTFILE" ]]; then
  echo "Error: hostfile not found: $HOSTFILE" >&2
  exit 1
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${SCRIPT_DIR}/$(date +%Y-%m-%d_%H-%M-%S)"
fi
if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="${SCRIPT_DIR}/${LOG_DIR}"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$LOG_DIR"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ ! -f "$WORK_DIR/$DIST_TRAIN" ]]; then
  echo "Error: per-node entry not found: ${WORK_DIR}/${DIST_TRAIN}" >&2
  echo "Refactor launch_multi2.sh → scripts/dist_train_megatron.sh first." >&2
  exit 1
fi

hostlist=$(grep -v '^#\|^$' "$HOSTFILE" | awk '{print $1}' | xargs)
read -ra ip_list <<< "$hostlist"
NNODES=${#ip_list[@]}
MASTER_ADDR=${ip_list[0]}

echo "number of nodes: ${NNODES}"
echo "master address: ${MASTER_ADDR}"
echo "work dir: ${WORK_DIR}"

find_free_port_on_host() {
  local host="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash <<'REMOTE_EOF' 2>/dev/null || true
for p in $(seq 20000 30000); do
  if command -v ss >/dev/null 2>&1; then
    if ! ss -ltn 2>/dev/null | grep -q ":${p} "; then
      echo "$p"
      exit 0
    fi
  elif command -v netstat >/dev/null 2>&1; then
    if ! netstat -ltn 2>/dev/null | grep -q ":${p} "; then
      echo "$p"
      exit 0
    fi
  else
    if ! (echo >/dev/tcp/127.0.0.1/${p}) 2>/dev/null; then
      echo "$p"
      exit 0
    fi
  fi
done
REMOTE_EOF
}

if [[ -z "$MAIN_PROCESS_PORT" ]]; then
  MAIN_PROCESS_PORT=$(find_free_port_on_host "$MASTER_ADDR")
  if [[ -n "$MAIN_PROCESS_PORT" ]]; then
    echo "Selected free main process port on ${MASTER_ADDR}: ${MAIN_PROCESS_PORT}"
  else
    echo "Warning: failed to detect free port on ${MASTER_ADDR}, using default"
  fi
fi

# Same loop as dist_run_fsdp8: one dist_train entry per node (not launch_multi2)
for host in ${ip_list[@]}; do
  echo "$host"
  ssh -f -n "$host" "bash -c 'cd \"${WORK_DIR}\"; WORLD_SIZE=${NNODES} RANK=${COUNT} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MAIN_PROCESS_PORT} TRAINING_STEPS=${TRAINING_STEPS} LOG_NAME=${LOG_NAME} ENABLE_TENSORBOARD=${ENABLE_TENSORBOARD} MUSA_PRETRAIN_ENTRY=${MUSA_PRETRAIN_ENTRY:-} ENABLE_SEQUENCE_PARALLEL=${ENABLE_SEQUENCE_PARALLEL:-} ENABLE_PROFILER=${ENABLE_PROFILER:-} PROFILER_FREQ=${PROFILER_FREQ:-} PROFILER_WARMUP_STEPS=${PROFILER_WARMUP_STEPS:-} PROFILER_PROFILE_MEMORY=${PROFILER_PROFILE_MEMORY:-} MUSA_LAUNCH_BLOCKING=${MUSA_LAUNCH_BLOCKING:-} PROFILE_STEP_START=${PROFILE_STEP_START:-} PROFILE_STEP_END=${PROFILE_STEP_END:-} USE_DEEPEP_ACE=${USE_DEEPEP_ACE:-} MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-} ENABLE_MOE_SHARED_EXPERT_OVERLAP=${ENABLE_MOE_SHARED_EXPERT_OVERLAP:-1} ENABLE_MOE_SE_LATE_ISSUE=${ENABLE_MOE_SE_LATE_ISSUE:-1} ENABLE_MOE_SHARED_EXPERT_EARLY=${ENABLE_MOE_SHARED_EXPERT_EARLY:-0} MOE_SE_DISPATCH_EVENT=${MOE_SE_DISPATCH_EVENT:-0} MUSA_NATIVE_ROPE=${MUSA_NATIVE_ROPE:-1} MUSA_FUSED_MLA_DOWN_PROJ=${MUSA_FUSED_MLA_DOWN_PROJ} MUSA_COMPACT_PERMUTE=${MUSA_COMPACT_PERMUTE} MATE_GROUPED_GEMM=${MATE_GROUPED_GEMM:-0} MATE_USE_MAIN_GRAD=${MATE_USE_MAIN_GRAD:-1} MATE_DEFER_DEEPEP_COUNTS=${MATE_DEFER_DEEPEP_COUNTS:-1} MATE_FLASH_ATTN=${MATE_FLASH_ATTN:-0} MATE_CACHE_MUBIN_DISPATCH=${MATE_CACHE_MUBIN_DISPATCH:-1} MUSA_CPU_AFFINITY=${MUSA_CPU_AFFINITY} MUSA_CPU_AFFINITY_MODE=${MUSA_CPU_AFFINITY_MODE} MUSA_CPU_AFFINITY_MAP=\"${MUSA_CPU_AFFINITY_MAP}\" bash ${DIST_TRAIN} > ${LOG_DIR}/log.${COUNT}.${host} 2>&1 &'"
  if [[ "$host" == "$MASTER_ADDR" ]]; then
    LOG_FILE="${LOG_DIR}/log.${COUNT}.${host}"
    echo "Waiting for master node log output..."
    while [[ ! -s "$LOG_FILE" ]]; do
      sleep 1
    done
    echo "Master node log has output, waiting 3 seconds..."
    sleep 3
  fi
  COUNT=$((COUNT + 1))
done
