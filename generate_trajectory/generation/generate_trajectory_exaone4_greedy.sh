#!/bin/bash
set -euo pipefail

# EXAONE4 wrapper for OpenCodeInstruct trajectory generation.
#
# Usage:
#   1) Edit json_files / save_path / model_path below
#   2) Run:
#        bash generate_trajectory/generation/generate_trajectory_exaone4_greedy.sh
#
# Notes:
# - Each file in json_files is launched on one visible GPU index.
# - Input files can be bucketed OpenCodeInstruct JSON files produced by
#   generate_trajectory/data/0_bucketing_opencodeinstruct.py

# ===== Config =====
json_files=(
    "/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/sample_prompts_small.json"
)

save_path="/workspace/exaone_workspace/JacobiForcing/tmp/exaone4_generated_trajectory_blk32"
log_dir="/workspace/exaone_workspace/JacobiForcing/tmp/exaone4_logs"

model_path="LGAI-EXAONE/EXAONE-4.0-1.2B"
n_token_seq_len=32
max_new_seq_len=1024
data_start_id=0
data_eos_id=500
dtype="bfloat16"
attn_implementation="flex_attention"
system_prompt="You are a helpful coding assistant."
num_gpus="${NUM_GPUS:-}"
batch_size="${BATCH_SIZE:-8}"
# ===== Config =====

mkdir -p "${save_path}"
mkdir -p "${log_dir}"

if [[ -z "${num_gpus}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        num_gpus="$(nvidia-smi -L | wc -l | tr -d ' ')"
    else
        num_gpus=1
    fi
fi

if [[ "${num_gpus}" -lt 1 ]]; then
    num_gpus=1
fi

echo "Using NUM_GPUS=${num_gpus}"

for i in "${!json_files[@]}"; do
    cuda_device=$(( i % num_gpus ))
    filename="${json_files[$i]}"
    base_name="$(basename "${filename}" .json)"
    log_file="${log_dir}/generate_trajectory_exaone4_${base_name}.log"

    echo "Device CUDA: ${cuda_device}"
    echo "Launching EXAONE4 trajectory generation on CUDA:${cuda_device} for file ${filename}"

    CUDA_VISIBLE_DEVICES=${cuda_device} python3 generate_trajectory/generation/generate_trajectory_exaone4_greedy.py \
        --filename "${filename}" \
        --model "${model_path}" \
        --n_token_seq_len "${n_token_seq_len}" \
        --max_new_seq_len "${max_new_seq_len}" \
        --data_bos_id "${data_start_id}" \
        --data_eos_id "${data_eos_id}" \
        --save_path "${save_path}" \
        --dtype "${dtype}" \
        --attn_implementation "${attn_implementation}" \
        --system_prompt "${system_prompt}" \
        --batch_size "${batch_size}" \
        > "${log_file}" 2>&1 &
done

wait
echo "All EXAONE4 trajectory generation processes completed."
