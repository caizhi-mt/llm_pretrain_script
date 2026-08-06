#!/bin/bash
# Per-node Megatron entry for K8s (JoyVideo dist_train pattern).
# Invoked by dist_run_megatron.sh with WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJ_DIR"

export HOSTFILE="${HOSTFILE:-${PROJ_DIR}/hostfile}"
export RANK="${RANK:-${NODE_RANK:-0}}"
export NNODES="${WORLD_SIZE:?WORLD_SIZE required}"
export MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR required}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# K8s PVC paths (jd-pvc22 + jd-ufs-pvc)
export TOKENIZER_PATH="${TOKENIZER_PATH:-${PROJ_DIR}/tokenizer}"
export DATA_PATH="${DATA_PATH:-/home/jd/wangkang/llm_pretrain/data/tkn_ds_the_pile}"
export SAVE_PATH="${SAVE_PATH:-/home/jd/wangkang/llm_pretrain/outputs}"
export LOG_OUTPUT="${LOG_OUTPUT:-/home/jd/wangkang/llm_pretrain/outputs/logs}"
export FOREGROUND="${FOREGROUND:-1}"

# JoyVideo dist_train.sh bootstrap（千卡 TCPStore rendezvous 需抬高 fd 上限）
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/jd/blake/mudnn0515/mudnn/lib:/usr/local/openmpi/lib:/usr/local/musa/lib:${LD_LIBRARY_PATH:-}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-900000}"
export ACCELERATOR_BACKEND="${ACCELERATOR_BACKEND:-musa}"
export MUSA_USERQ="${MUSA_USERQ:-1}"
export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_MCCL_USE_DEFAULT_STREAM="${TORCH_MCCL_USE_DEFAULT_STREAM:-1}"
export TORCH_MCCL_AVOID_RECORD_STREAMS="${TORCH_MCCL_AVOID_RECORD_STREAMS:-1}"
export TOKENIZERS_PARALLELISM=false
MUSA_PRETRAIN_ENTRY="${MUSA_PRETRAIN_ENTRY:-musa_pretrain_ws128.sh}"
if [[ "${MUSA_PRETRAIN_ENTRY}" != *flex_deepep* ]]; then
  export MCCL_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME:-bond0}"
  export MCCL_CROSS_NIC="${MCCL_CROSS_NIC:-0}"
fi
# 与 JoyVideo 千卡一致；musa_pretrain_multi2.sh 中 RoCE 默认值可被此处覆盖

ulimit -n 524288
ulimit -s unlimited
ulimit -c unlimited

echo "dist_train_megatron: RANK=${RANK} NNODES=${NNODES} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  SAVE_PATH=${SAVE_PATH}"
echo "  ulimit -n=$(ulimit -n)"

echo "  MUSA_PRETRAIN_ENTRY=${MUSA_PRETRAIN_ENTRY}"

exec bash "${PROJ_DIR}/${MUSA_PRETRAIN_ENTRY}"
