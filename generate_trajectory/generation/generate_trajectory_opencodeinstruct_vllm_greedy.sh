#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-nota-ai/Solar-Open-100B-Nota-FP8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
SPLIT_FILE="${SPLIT_FILE:-}"
SAVE_PATH="${SAVE_PATH:-}"
STEP="${STEP:-split_a}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-solar}"
MAX_NEW_SEQ_LEN="${MAX_NEW_SEQ_LEN:-1024}"
SEED="${SEED:-42}"
GPUS="${GPUS:-4 5 6 7}"
CACHE_ROOT="${CACHE_ROOT:-${PWD}/.cache/vllm_runtime}"
MAX_RECORDS="${MAX_RECORDS:--1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_ACTIVE_PROMPTS="${MAX_ACTIVE_PROMPTS:-16}"
DTYPE="${DTYPE:-bfloat16}"
LOGPROBS_MODE="${LOGPROBS_MODE:-processed_logprobs}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${MODEL_PATH}" || -z "${TOKENIZER_PATH}" ]]; then
    echo "MODEL_PATH and TOKENIZER_PATH are required." >&2
    exit 1
fi
mkdir -p "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/triton"

read -r -a GPU_LIST <<< "${GPUS}"
TP_SIZE="${#GPU_LIST[@]}"
if [[ "${TP_SIZE}" -eq 0 ]]; then
    echo "No GPUs configured via GPUS." >&2
    exit 1
fi

GPU_CSV="$(IFS=,; echo "${GPU_LIST[*]}")"

extra_args=()
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" || "${DISABLE_CUSTOM_ALL_REDUCE}" == "true" ]]; then
    extra_args+=(--disable_custom_all_reduce)
else
    extra_args+=(--no-disable_custom_all_reduce)
fi

export CUDA_VISIBLE_DEVICES="${GPU_CSV}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export VLLM_ATTENTION_BACKEND

if [[ -n "${PYTHON_BIN}" ]]; then
    read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif [[ -x "${PWD}/../.venv-vllm/bin/python" ]]; then
    PYTHON_CMD=("${PWD}/../.venv-vllm/bin/python")
elif [[ -x "${PWD}/../vllm/.venv/bin/python" ]]; then
    PYTHON_CMD=("${PWD}/../vllm/.venv/bin/python")
elif [[ -f "${PWD}/../vllm/pyproject.toml" ]]; then
    PYTHON_CMD=(uv run --directory "${PWD}/../vllm" python)
else
    PYTHON_CMD=(uv run python)
fi

run_step() {
    local step_name="$1"
    local n_token_seq_len="$2"
    local split_file="$3"
    local save_path="$4"
    local log_dir="${LOG_DIR:-${save_path%/}/logs}"
    local log_file="${log_dir}/vllm_tp${TP_SIZE}_ep${TP_SIZE}.log"

    mkdir -p "${save_path}" "${log_dir}"

    echo "[vllm-traj] model=${MODEL_PATH}"
    echo "[vllm-traj] tokenizer=${TOKENIZER_PATH}"
    echo "[vllm-traj] split_file=${split_file}"
    echo "[vllm-traj] save_path=${save_path}"
    echo "[vllm-traj] log_file=${log_file}"
    echo "[vllm-traj] VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"
    echo "[vllm-traj] chat_template_mode=${CHAT_TEMPLATE_MODE}"
    echo "[vllm-traj] n_token_seq_len=${n_token_seq_len} max_new_seq_len=${MAX_NEW_SEQ_LEN}"
    echo "[vllm-traj] step=${step_name} seed=${SEED} max_records=${MAX_RECORDS}"
    echo "[vllm-traj] physical GPUs=${GPU_CSV} TP=${TP_SIZE} EP=enabled"
    echo "[vllm-traj] cache_root=${CACHE_ROOT}"
    echo "[vllm-traj] attention_backend=${VLLM_ATTENTION_BACKEND}"
    echo "[vllm-traj] disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE}"
    echo "[vllm-traj] max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} max_active_prompts=${MAX_ACTIVE_PROMPTS}"

    "${PYTHON_CMD[@]}" generate_trajectory/generation/generate_trajectory_opencodeinstruct_vllm_greedy.py \
        --filename "${split_file}" \
        --model "${MODEL_PATH}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --n_token_seq_len "${n_token_seq_len}" \
        --max_new_seq_len "${MAX_NEW_SEQ_LEN}" \
        --data_bos_id 0 \
        --data_eos_id "${MAX_RECORDS}" \
        --seed "${SEED}" \
        --chat_template_mode "${CHAT_TEMPLATE_MODE}" \
        --save_path "${save_path}" \
        --tensor_parallel_size "${TP_SIZE}" \
        --enable_expert_parallel \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --max_active_prompts "${MAX_ACTIVE_PROMPTS}" \
        --dtype "${DTYPE}" \
        --logprobs_mode "${LOGPROBS_MODE}" \
        "${extra_args[@]}" \
        > "${log_file}" 2>&1

    echo "vLLM TP${TP_SIZE}/EP${TP_SIZE} trajectory generation completed for ${step_name}."
    echo "[vllm-traj] detailed log saved to ${log_file}"
}

case "${STEP}" in
    split_a)
        run_step \
            "split_a" \
            16 \
            "${SPLIT_FILE:-runs/solar_open_100b_opencode_2step/opencodeinstruct_solar_splits/split_a.json}" \
            "${SAVE_PATH:-runs/solar_open_100b_opencode_2step/pipeline_vllm/step1_split_a_n16w16/traj_shards}"
        ;;
    split_b)
        run_step \
            "split_b" \
            32 \
            "${SPLIT_FILE:-runs/solar_open_100b_opencode_2step/opencodeinstruct_solar_splits/split_b.json}" \
            "${SAVE_PATH:-runs/solar_open_100b_opencode_2step/pipeline_vllm/step2_split_b_n32w8/traj_shards}"
        ;;
    all)
        run_step \
            "split_a" \
            16 \
            "runs/solar_open_100b_opencode_2step/opencodeinstruct_solar_splits/split_a.json" \
            "runs/solar_open_100b_opencode_2step/pipeline_vllm/step1_split_a_n16w16/traj_shards"
        run_step \
            "split_b" \
            32 \
            "runs/solar_open_100b_opencode_2step/opencodeinstruct_solar_splits/split_b.json" \
            "runs/solar_open_100b_opencode_2step/pipeline_vllm/step2_split_b_n32w8/traj_shards"
        ;;
    *)
        echo "STEP must be one of: split_a, split_b, all" >&2
        exit 1
        ;;
esac
