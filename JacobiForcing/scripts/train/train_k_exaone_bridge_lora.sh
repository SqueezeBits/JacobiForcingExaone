#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:?Set MODEL_ID to the K-EXAONE checkpoint or HF id}"
PACKED_DATA="${PACKED_DATA:?Set PACKED_DATA to a packed training JSONL path}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/exaone_workspace/JacobiForcing/tmp/k_exaone_bridge_lora}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-4}"
MICRO_BS="${MICRO_BS:-1}"
GLOBAL_BS="${GLOBAL_BS:-4}"
MAX_STEPS="${MAX_STEPS:-10}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

cat <<EOF
K-EXAONE Megatron Bridge LoRA launcher template
model_id=${MODEL_ID}
packed_data=${PACKED_DATA}
output_dir=${OUTPUT_DIR}
tp=${TP_SIZE} pp=${PP_SIZE} ep=${EP_SIZE}
micro_bs=${MICRO_BS} global_bs=${GLOBAL_BS}
max_steps=${MAX_STEPS} lr=${LR}
lora_rank=${LORA_RANK} lora_alpha=${LORA_ALPHA} lora_dropout=${LORA_DROPOUT}

Recommended next step:
  1. Install pixi env
  2. Confirm megatron-bridge + nemo are importable
  3. Replace this template with the project-specific bridge trainer invocation
EOF
