#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:?Set MODEL_ID to the K-EXAONE checkpoint or HF id}"
PACKED_DATA="${PACKED_DATA:?Set PACKED_DATA to a packed training JSONL path}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/exaone_workspace/JacobiForcing/tmp/k_exaone_bridge_full}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-4}"
MICRO_BS="${MICRO_BS:-1}"
GLOBAL_BS="${GLOBAL_BS:-4}"
MAX_STEPS="${MAX_STEPS:-1}"
LR="${LR:-5e-6}"

cat <<EOF
K-EXAONE Megatron Bridge / NeMo full fine-tuning launcher template
model_id=${MODEL_ID}
packed_data=${PACKED_DATA}
output_dir=${OUTPUT_DIR}
tp=${TP_SIZE} pp=${PP_SIZE} ep=${EP_SIZE}
micro_bs=${MICRO_BS} global_bs=${GLOBAL_BS}
max_steps=${MAX_STEPS} lr=${LR}

NOTE:
  Full FT on 1 node / 4x B200 for a 236B MoE model is intended as a viability
  check only. Expect memory pressure and parallelism tuning work.
EOF
