#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-4.0-1.2B}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
WINDOW_SIZE="${WINDOW_SIZE:-4}"
MAX_NEW_SEQ_LEN="${MAX_NEW_SEQ_LEN:-8}"
PROMPT_FILE="${PROMPT_FILE:-}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT:-/workspace/exaone_workspace/JacobiForcing/tmp/exaone4_tiny_trainset}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -z "${PROMPT_FILE}" ]]; then
  PROMPT_FILE="${SCRIPT_DIR}/../tool/sample_prompts_small.json"
fi

mkdir -p "${OUT_DIR}"

TRAJ_DIR="${OUT_DIR}/trajectory"
PACKED_JSONL="${OUT_DIR}/packed.jsonl"

python "${REPO_ROOT}/generate_trajectory/generation/generate_trajectory_exaone4_greedy.py" \
  --model "${MODEL_ID}" \
  --filename "${PROMPT_FILE}" \
  --save_path "${TRAJ_DIR}" \
  --n_token_seq_len "${BLOCK_SIZE}" \
  --max_new_seq_len "${MAX_NEW_SEQ_LEN}" \
  --data_bos_id 0 \
  --data_eos_id 9999

TRAJ_FILE="$(ls -1 "${TRAJ_DIR}"/*.json | head -n 1)"

python "${REPO_ROOT}/generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py" \
  --input_path "${TRAJ_FILE}" \
  --output_path "${PACKED_JSONL}" \
  --n_token_seq_length "${BLOCK_SIZE}" \
  --window_size "${WINDOW_SIZE}" \
  --min_noisy_ratio 0 \
  --max_noisy_ratio 1.0 \
  --strategy progressive \
  --single-process \
  --num-workers 1

echo "trajectory_file=${TRAJ_FILE}"
echo "packed_file=${PACKED_JSONL}"
