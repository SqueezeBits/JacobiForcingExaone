#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-4.0-1.2B}"
DATA_PATH="${DATA_PATH:-/workspace/exaone_workspace/JacobiForcing/tmp/exaone4_tiny_trainset/packed.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/exaone_workspace/JacobiForcing/tmp/exaone4_tiny_pilot_ckpt}"
DTYPE="${DTYPE:-bfloat16}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-4}"
SAVE_EVERY="${SAVE_EVERY:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/train_exaone4_tiny_pilot.py" \
  --model-id "${MODEL_ID}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --dtype "${DTYPE}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --epochs "${EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  "$@"
