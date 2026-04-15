#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
SPLIT_FILE="${SPLIT_FILE:-}"
SAVE_PATH="${SAVE_PATH:-}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-solar}"
N_TOKEN_SEQ_LEN="${N_TOKEN_SEQ_LEN:-16}"
MAX_NEW_SEQ_LEN="${MAX_NEW_SEQ_LEN:-1024}"
SEED="${SEED:-42}"
GPUS="${GPUS:-4 5 6 7}"
LOG_DIR="${LOG_DIR:-${SAVE_PATH%/}/logs}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_RECORDS="${MAX_RECORDS:--1}"

if [[ -z "${MODEL_PATH}" || -z "${TOKENIZER_PATH}" || -z "${SPLIT_FILE}" || -z "${SAVE_PATH}" ]]; then
    echo "MODEL_PATH, TOKENIZER_PATH, SPLIT_FILE, and SAVE_PATH are required." >&2
    exit 1
fi

mkdir -p "${SAVE_PATH}" "${LOG_DIR}"

read -r -a GPU_LIST <<< "${GPUS}"
TP_SIZE="${#GPU_LIST[@]}"
if [[ "${TP_SIZE}" -eq 0 ]]; then
    echo "No GPUs configured via GPUS." >&2
    exit 1
fi

GPU_CSV="$(IFS=,; echo "${GPU_LIST[*]}")"
LOG_FILE="${LOG_DIR}/tp${TP_SIZE}.log"

echo "[traj] model=${MODEL_PATH}"
echo "[traj] tokenizer=${TOKENIZER_PATH}"
echo "[traj] split_file=${SPLIT_FILE}"
echo "[traj] save_path=${SAVE_PATH}"
echo "[traj] log_file=${LOG_FILE}"
echo "[traj] chat_template_mode=${CHAT_TEMPLATE_MODE}"
echo "[traj] n_token_seq_len=${N_TOKEN_SEQ_LEN} max_new_seq_len=${MAX_NEW_SEQ_LEN}"
echo "[traj] seed=${SEED} max_records=${MAX_RECORDS}"
echo "Launching TP${TP_SIZE} trajectory generation for ${SPLIT_FILE}"
echo "Using physical GPUs: ${GPU_CSV}"

CUDA_VISIBLE_DEVICES="${GPU_CSV}" \
"$(pwd)/.venv/bin/torchrun" \
    --standalone \
    --nproc-per-node 1 \
    --master-port "${MASTER_PORT}" \
    generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.py \
    --filename "${SPLIT_FILE}" \
    --model "${MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --n_token_seq_len "${N_TOKEN_SEQ_LEN}" \
    --max_new_seq_len "${MAX_NEW_SEQ_LEN}" \
    --data_bos_id 0 \
    --data_eos_id "${MAX_RECORDS}" \
    --seed "${SEED}" \
    --chat_template_mode "${CHAT_TEMPLATE_MODE}" \
    --save_path "${SAVE_PATH}" \
    > "${LOG_FILE}" 2>&1

echo "TP${TP_SIZE} trajectory generation completed."
echo "[traj] detailed log saved to ${LOG_FILE}"
