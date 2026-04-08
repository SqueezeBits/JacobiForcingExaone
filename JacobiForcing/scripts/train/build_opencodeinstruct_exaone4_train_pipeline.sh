#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline:
#   1) Bucket OpenCodeInstruct JSONL files
#   2) Generate EXAONE4 Jacobi trajectories from selected bucket files
#   3) Pack them into Jacobi Forcing training JSONL
#
# Minimal usage:
#   source /opt/miniforge3/etc/profile.d/conda.sh
#   conda activate jacobi
#   OPENCODE_INPUT_DIR=/path/to/opencodeinstruct_jsonl_dir \
#   bash JacobiForcing/scripts/train/build_opencodeinstruct_exaone4_train_pipeline.sh
#
# Resume examples:
#   START_STAGE=3 bash .../build_opencodeinstruct_exaone4_train_pipeline.sh
#     -> skip bucketing and trajectory generation, pack existing trajectory/*.json
#   START_STAGE=2 bash .../build_opencodeinstruct_exaone4_train_pipeline.sh
#     -> reuse existing bucketed files, regenerate trajectories and packed files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

OPENCODE_INPUT_DIR="${OPENCODE_INPUT_DIR:-}"
OUT_ROOT="${OUT_ROOT:-/workspace/exaone_workspace/JacobiForcing/tmp/opencodeinstruct_exaone4_pipeline}"
MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-4.0-1.2B}"

BUCKET_SIZE="${BUCKET_SIZE:-5000}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
WINDOW_SIZE="${WINDOW_SIZE:-16}"
MAX_NEW_SEQ_LEN="${MAX_NEW_SEQ_LEN:-1024}"
DATA_BOS_ID="${DATA_BOS_ID:-0}"
DATA_EOS_ID="${DATA_EOS_ID:-500}"
NUM_BUCKETS="${NUM_BUCKETS:-4}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-flex_attention}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful coding assistant.}"
NUM_GPUS="${NUM_GPUS:-}"
START_STAGE="${START_STAGE:-1}"
OVERWRITE_PACKED="${OVERWRITE_PACKED:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ "${START_STAGE}" -le 1 && -z "${OPENCODE_INPUT_DIR}" ]]; then
  echo "ERROR: set OPENCODE_INPUT_DIR to a directory containing OpenCodeInstruct JSONL files." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
BUCKET_DIR="${OUT_ROOT}/bucketed"
TRAJ_DIR="${OUT_ROOT}/trajectory"
PACKED_DIR="${OUT_ROOT}/packed"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${BUCKET_DIR}" "${TRAJ_DIR}" "${PACKED_DIR}" "${LOG_DIR}"

if [[ -z "${NUM_GPUS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NUM_GPUS=1
  fi
fi

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  NUM_GPUS=1
fi

if [[ "${START_STAGE}" -le 1 ]]; then
  echo "[1/3] Bucketing OpenCodeInstruct -> ${BUCKET_DIR}"
  python "${REPO_ROOT}/generate_trajectory/data/0_bucketing_opencodeinstruct.py" \
    --input_path "${OPENCODE_INPUT_DIR}" \
    --output_path "${BUCKET_DIR}" \
    --tokenizer_path "${MODEL_ID}" \
    --bucket_size "${BUCKET_SIZE}"
else
  echo "[1/3] Skipped bucketing because START_STAGE=${START_STAGE}"
fi

mapfile -t SELECTED_BUCKETS < <(find "${BUCKET_DIR}" -maxdepth 1 -type f -name 'bucket_*.json' | sort | head -n "${NUM_BUCKETS}" || true)

if [[ "${#SELECTED_BUCKETS[@]}" -eq 0 ]]; then
  echo "ERROR: no bucket_*.json files found under ${BUCKET_DIR}" >&2
  exit 1
fi

echo "[info] Selected bucket files:"
for f in "${SELECTED_BUCKETS[@]}"; do
  echo "  - ${f}"
done

if [[ "${START_STAGE}" -le 2 ]]; then
  echo "[2/3] Generating EXAONE4 trajectories -> ${TRAJ_DIR}"
  echo "[info] NUM_GPUS=${NUM_GPUS}"
  for idx in "${!SELECTED_BUCKETS[@]}"; do
    cuda_device=$(( idx % NUM_GPUS ))
    filename="${SELECTED_BUCKETS[$idx]}"
    base_name="$(basename "${filename}" .json)"
    log_file="${LOG_DIR}/trajectory_${base_name}.log"

    echo "  [bucket ${idx}] CUDA_VISIBLE_DEVICES=${cuda_device} file=${filename}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" python "${REPO_ROOT}/generate_trajectory/generation/generate_trajectory_exaone4_greedy.py" \
      --model "${MODEL_ID}" \
      --filename "${filename}" \
      --save_path "${TRAJ_DIR}" \
      --n_token_seq_len "${BLOCK_SIZE}" \
      --max_new_seq_len "${MAX_NEW_SEQ_LEN}" \
      --data_bos_id "${DATA_BOS_ID}" \
      --data_eos_id "${DATA_EOS_ID}" \
      --dtype "${DTYPE}" \
      --attn_implementation "${ATTN_IMPL}" \
      --system_prompt "${SYSTEM_PROMPT}" \
      --batch_size "${BATCH_SIZE}" \
      > "${log_file}" 2>&1
  done
else
  echo "[2/3] Skipped trajectory generation because START_STAGE=${START_STAGE}"
fi

echo "[3/3] Packing trajectory files -> ${PACKED_DIR}"
shopt -s nullglob
traj_files=( "${TRAJ_DIR}"/*.json )
shopt -u nullglob
if [[ "${#traj_files[@]}" -eq 0 ]]; then
  echo "ERROR: no trajectory JSON files found under ${TRAJ_DIR}" >&2
  exit 1
fi

for traj_file in "${TRAJ_DIR}"/*.json; do
  base_name="$(basename "${traj_file}" .json)"
  packed_file="${PACKED_DIR}/${base_name}.jsonl"
  if [[ "${OVERWRITE_PACKED}" != "1" && -f "${packed_file}" ]]; then
    echo "  [skip-pack] ${packed_file} already exists"
    continue
  fi
  echo "  [pack] ${traj_file} -> ${packed_file}"
  python "${REPO_ROOT}/generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py" \
    --input_path "${traj_file}" \
    --output_path "${packed_file}" \
    --n_token_seq_length "${BLOCK_SIZE}" \
    --window_size "${WINDOW_SIZE}" \
    --min_noisy_ratio 0 \
    --max_noisy_ratio 1.0 \
    --strategy progressive \
    --single-process \
    --num-workers 1
done

echo
echo "Done."
echo "bucket_dir=${BUCKET_DIR}"
echo "trajectory_dir=${TRAJ_DIR}"
echo "packed_dir=${PACKED_DIR}"
echo "log_dir=${LOG_DIR}"
