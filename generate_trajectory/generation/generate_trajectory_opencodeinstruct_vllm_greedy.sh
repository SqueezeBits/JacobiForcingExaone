#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-nota-ai/Solar-Open-100B-Nota-FP8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
SPLIT_FILE="${SPLIT_FILE:-}"
SAVE_PATH="${SAVE_PATH:-}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-solar}"
N_TOKEN_SEQ_LEN="${N_TOKEN_SEQ_LEN:-16}"
MAX_NEW_SEQ_LEN="${MAX_NEW_SEQ_LEN:-1024}"
SEED="${SEED:-42}"
GPUS="${GPUS:-4 5 6 7}"
LOG_DIR="${LOG_DIR:-${SAVE_PATH%/}/logs}"
CACHE_ROOT="${CACHE_ROOT:-${PWD}/.cache/vllm_runtime}"
MAX_RECORDS="${MAX_RECORDS:--1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
FALLBACK_MODE="${FALLBACK_MODE:-auto}"
FALLBACK_CALIBRATION_BLOCKS="${FALLBACK_CALIBRATION_BLOCKS:-2}"
INCLUDE_DIAGNOSTICS="${INCLUDE_DIAGNOSTICS:-0}"
VLLM_REPO="${VLLM_REPO:-../vllm}"
DTYPE="${DTYPE:-bfloat16}"
LOGPROBS_MODE="${LOGPROBS_MODE:-processed_logprobs}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${MODEL_PATH}" || -z "${TOKENIZER_PATH}" || -z "${SPLIT_FILE}" || -z "${SAVE_PATH}" ]]; then
    echo "MODEL_PATH, TOKENIZER_PATH, SPLIT_FILE, and SAVE_PATH are required." >&2
    exit 1
fi

mkdir -p "${SAVE_PATH}" "${LOG_DIR}" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/triton"

read -r -a GPU_LIST <<< "${GPUS}"
TP_SIZE="${#GPU_LIST[@]}"
if [[ "${TP_SIZE}" -eq 0 ]]; then
    echo "No GPUs configured via GPUS." >&2
    exit 1
fi

GPU_CSV="$(IFS=,; echo "${GPU_LIST[*]}")"
LOG_FILE="${LOG_DIR}/vllm_tp${TP_SIZE}_ep${TP_SIZE}.log"

extra_args=()
if [[ "${INCLUDE_DIAGNOSTICS}" == "1" || "${INCLUDE_DIAGNOSTICS}" == "true" ]]; then
    extra_args+=(--include_diagnostics)
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" || "${DISABLE_CUSTOM_ALL_REDUCE}" == "true" ]]; then
    extra_args+=(--disable_custom_all_reduce)
else
    extra_args+=(--no-disable_custom_all_reduce)
fi

# Use the cloned UpstageAI/vllm checkout by default, while still allowing an
# already-installed vLLM environment to take over by setting VLLM_REPO="".
if [[ -n "${VLLM_REPO}" ]]; then
    export PYTHONPATH="${VLLM_REPO}:${PYTHONPATH:-}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_CSV}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export VLLM_ATTENTION_BACKEND

echo "[vllm-traj] model=${MODEL_PATH}"
echo "[vllm-traj] tokenizer=${TOKENIZER_PATH}"
echo "[vllm-traj] split_file=${SPLIT_FILE}"
echo "[vllm-traj] save_path=${SAVE_PATH}"
echo "[vllm-traj] log_file=${LOG_FILE}"
echo "[vllm-traj] chat_template_mode=${CHAT_TEMPLATE_MODE}"
echo "[vllm-traj] n_token_seq_len=${N_TOKEN_SEQ_LEN} max_new_seq_len=${MAX_NEW_SEQ_LEN}"
echo "[vllm-traj] seed=${SEED} max_records=${MAX_RECORDS} fallback=${FALLBACK_MODE}"
echo "[vllm-traj] physical GPUs=${GPU_CSV} TP=${TP_SIZE} EP=enabled"
echo "[vllm-traj] cache_root=${CACHE_ROOT}"
echo "[vllm-traj] attention_backend=${VLLM_ATTENTION_BACKEND}"
echo "[vllm-traj] disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE}"

if [[ -n "${PYTHON_BIN}" ]]; then
    read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
elif [[ -n "${VLLM_REPO}" && -x "$(dirname "${VLLM_REPO}")/.venv-vllm/bin/python" ]]; then
    # Prefer the workspace-level vLLM virtualenv when present. `uv run --project`
    # may otherwise reuse the caller workspace environment and pull incompatible
    # Hugging Face/kernel packages.
    PYTHON_CMD=("$(dirname "${VLLM_REPO}")/.venv-vllm/bin/python")
elif [[ -n "${VLLM_REPO}" && -x "${VLLM_REPO}/.venv/bin/python" ]]; then
    PYTHON_CMD=("${VLLM_REPO}/.venv/bin/python")
elif [[ -n "${VLLM_REPO}" && -f "${VLLM_REPO}/pyproject.toml" ]]; then
    PYTHON_CMD=(uv run --directory "${VLLM_REPO}" python)
else
    PYTHON_CMD=(uv run python)
fi

"${PYTHON_CMD[@]}" generate_trajectory/generation/generate_trajectory_opencodeinstruct_vllm_greedy.py \
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
    --tensor_parallel_size "${TP_SIZE}" \
    --enable_expert_parallel \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --dtype "${DTYPE}" \
    --logprobs_mode "${LOGPROBS_MODE}" \
    --fallback_mode "${FALLBACK_MODE}" \
    --fallback_calibration_blocks "${FALLBACK_CALIBRATION_BLOCKS}" \
    --vllm_repo "${VLLM_REPO}" \
    "${extra_args[@]}" \
    > "${LOG_FILE}" 2>&1

echo "vLLM TP${TP_SIZE}/EP${TP_SIZE} trajectory generation completed."
echo "[vllm-traj] detailed log saved to ${LOG_FILE}"
