#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-k-exaone}"
INPUT_BUCKET="${INPUT_BUCKET:?Set INPUT_BUCKET to a bucket json/jsonl file}"
OUT_ROOT="${OUT_ROOT:-/workspace/exaone_workspace/JacobiForcing/tmp/k_exaone_vllm_trainset}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
WINDOW_SIZE="${WINDOW_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TP_SIZE="${TP_SIZE:-1}"
REFINEMENT_STEPS="${REFINEMENT_STEPS:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${OUT_ROOT}"
TRAJ_JSON="${OUT_ROOT}/trajectory.json"
PACKED_JSONL="${OUT_ROOT}/packed.jsonl"

python "${SCRIPT_DIR}/generate_k_exaone_vllm_dataset.py" \
  --model-id "${MODEL_ID}" \
  --input-file "${INPUT_BUCKET}" \
  --output-file "${TRAJ_JSON}" \
  --block-size "${BLOCK_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-samples "${MAX_SAMPLES}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --refinement-steps "${REFINEMENT_STEPS}"

python "${REPO_ROOT}/generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py" \
  --input_path "${TRAJ_JSON}" \
  --output_path "${PACKED_JSONL}" \
  --n_token_seq_length "${BLOCK_SIZE}" \
  --window_size "${WINDOW_SIZE}" \
  --min_noisy_ratio 0 \
  --max_noisy_ratio 1.0 \
  --strategy progressive \
  --single-process \
  --num-workers 1

echo "trajectory_file=${TRAJ_JSON}"
echo "packed_file=${PACKED_JSONL}"
