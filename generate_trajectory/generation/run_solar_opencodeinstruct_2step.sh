#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
SPLIT_DIR="${SPLIT_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
STEP="${STEP:-all}"
GPUS="${GPUS:-4 5 6 7}"
MAX_NEW_SEQ_LEN="${MAX_NEW_SEQ_LEN:-1024}"
SEED="${SEED:-42}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-solar}"
MAX_RECORDS="${MAX_RECORDS:--1}"

if [[ -z "${MODEL_PATH}" || -z "${TOKENIZER_PATH}" || -z "${SPLIT_DIR}" || -z "${OUTPUT_ROOT}" ]]; then
    echo "MODEL_PATH, TOKENIZER_PATH, SPLIT_DIR, and OUTPUT_ROOT are required." >&2
    exit 1
fi

SPLIT_A_FILE="${SPLIT_A_FILE:-${SPLIT_DIR%/}/split_a.json}"
SPLIT_B_FILE="${SPLIT_B_FILE:-${SPLIT_DIR%/}/split_b.json}"

mkdir -p "${OUTPUT_ROOT}"

run_step() {
    local split_name="$1"
    local split_file="$2"
    local n_token_seq_len="$3"
    local window_size="$4"
    local step_dir="$5"
    local traj_dir="${step_dir}/traj_shards"
    local log_dir="${step_dir}/logs"
    local merged_dir="${step_dir}/merged"
    local packed_dir="${step_dir}/packed"
    local merged_file="${merged_dir}/trajectory_${split_name}_n${n_token_seq_len}w${window_size}.jsonl"
    local packed_file="${packed_dir}/packed_${split_name}_n${n_token_seq_len}w${window_size}.jsonl"

    mkdir -p "${traj_dir}" "${log_dir}" "${merged_dir}" "${packed_dir}"
    rm -f "${traj_dir}"/*.json "${traj_dir}"/*.jsonl "${merged_file}" "${packed_file}"

    echo "[step:${split_name}] start"
    echo "[step:${split_name}] split_file=${split_file}"
    echo "[step:${split_name}] step_dir=${step_dir}"
    echo "[step:${split_name}] traj_dir=${traj_dir}"
    echo "[step:${split_name}] merged_file=${merged_file}"
    echo "[step:${split_name}] packed_file=${packed_file}"
    echo "[step:${split_name}] n_token_seq_len=${n_token_seq_len} window_size=${window_size} max_records=${MAX_RECORDS}"

    LOG_DIR="${log_dir}" \
    MODEL_PATH="${MODEL_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    SPLIT_FILE="${split_file}" \
    SAVE_PATH="${traj_dir}" \
    CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE}" \
    N_TOKEN_SEQ_LEN="${n_token_seq_len}" \
    MAX_NEW_SEQ_LEN="${MAX_NEW_SEQ_LEN}" \
    SEED="${SEED}" \
    GPUS="${GPUS}" \
    MAX_RECORDS="${MAX_RECORDS}" \
    bash generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.sh

    echo "[step:${split_name}] trajectory generation finished"
    echo "[step:${split_name}] merging shard json files"

    uv run python generate_trajectory/data/tool_merge_single_bucket_data.py \
        --input_dir "${traj_dir}" \
        --output "${merged_file}" \
        --pattern "*.json"

    echo "[step:${split_name}] merge finished"
    echo "[step:${split_name}] packing progressive noise training data"

    uv run python generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py \
        --input_path "${merged_file}" \
        --output_path "${packed_file}" \
        --n_token_seq_length "${n_token_seq_len}" \
        --window_size "${window_size}" \
        --min_noisy_ratio 0 \
        --max_noisy_ratio 1.0 \
        --strategy progressive \
        --seed "${SEED}"

    echo "[step:${split_name}] packing finished"
    echo "[step:${split_name}] done"
}

case "${STEP}" in
    all)
        run_step "split_a" "${SPLIT_A_FILE}" 16 16 "${OUTPUT_ROOT%/}/step1_split_a_n16w16"
        run_step "split_b" "${SPLIT_B_FILE}" 32 8 "${OUTPUT_ROOT%/}/step2_split_b_n32w8"
        ;;
    split_a)
        run_step "split_a" "${SPLIT_A_FILE}" 16 16 "${OUTPUT_ROOT%/}/step1_split_a_n16w16"
        ;;
    split_b)
        run_step "split_b" "${SPLIT_B_FILE}" 32 8 "${OUTPUT_ROOT%/}/step2_split_b_n32w8"
        ;;
    *)
        echo "STEP must be one of: all, split_a, split_b" >&2
        exit 1
        ;;
esac
