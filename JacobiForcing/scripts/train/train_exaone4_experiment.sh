#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-4.0-1.2B}"
DATA_PATH="${DATA_PATH:-/workspace/exaone_workspace/JacobiForcing/tmp/exaone4_tiny_trainset/packed.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/exaone_workspace/JacobiForcing/tmp/exaone4_experiment_run}"
DTYPE="${DTYPE:-bfloat16}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-20}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LOG_EVERY="${LOG_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SAMPLE_EVERY="${SAMPLE_EVERY:-5}"
SAMPLE_MAX_NEW_TOKENS="${SAMPLE_MAX_NEW_TOKENS:-64}"
SEED="${SEED:-7}"
RESUME_FROM="${RESUME_FROM:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/train_exaone4_experiment.py" \
  --model-id "${MODEL_ID}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --dtype "${DTYPE}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --epochs "${EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --lr-scheduler "${LR_SCHEDULER}" \
  --log-every "${LOG_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  --sample-every "${SAMPLE_EVERY}" \
  --sample-max-new-tokens "${SAMPLE_MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  ${RESUME_FROM:+--resume-from "${RESUME_FROM}"} \
  "$@"
