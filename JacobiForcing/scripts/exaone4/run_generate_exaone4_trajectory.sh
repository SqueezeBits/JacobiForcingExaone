#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-4.0-1.2B}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_NEW_SEQ_LEN="${MAX_NEW_SEQ_LEN:-128}"
INPUT_FILE="${INPUT_FILE:?Set INPUT_FILE to a JSON/JSONL prompt file}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a destination directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

python "${REPO_ROOT}/generate_trajectory/generation/generate_trajectory_exaone4_greedy.py" \
  --model "${MODEL_ID}" \
  --filename "${INPUT_FILE}" \
  --save_path "${OUTPUT_DIR}" \
  --n_token_seq_len "${BLOCK_SIZE}" \
  --max_new_seq_len "${MAX_NEW_SEQ_LEN}" \
  "$@"
