# Data Generation

이 문서는 현재 리포에서 `Solar-Open-100B`용 JacobiForcing 데이터를 어떻게 준비하는지, 실제로 어떤 파일을 실행하는지, 각 파일이 무슨 역할을 하는지 쉽게 설명한다.

기준 경로는 리포 루트다. 아래 명령들은 모두 리포 루트에서 실행한다고 가정한다.

## 개요

전체 흐름은 4단계다.

1. `OpenCodeInstruct`에서 샘플을 뽑고 Solar tokenizer 기준으로 길이 버킷을 만든다.
2. 버킷 결과에서 총 40K를 고정 seed로 나누어 `split_a`와 `split_b`를 만든다.
3. `split_a`에 대해 trajectory를 만들고 packed training data를 만든다.
4. `split_b`에 대해 같은 작업을 반복한다.

현재 설정은 다음과 같다.

- 데이터셋: `nvidia/OpenCodeInstruct`
- 모델/토크나이저: `upstage/Solar-Open-100B`
- step1: `n=16`, `w=16`
- step2: `n=32`, `w=8`
- GPU 사용: 물리 GPU `4,5,6,7`
- 분산 방식: `TP4`만 사용, `EP`는 사용하지 않음

## 실행 전 준비

리포 루트에서 아래를 먼저 실행한다.

```bash
# run from the repository root
uv sync
```

그다음 아래 환경변수를 쓴다.

```bash
export OCI_INPUT_PATH="nvidia/OpenCodeInstruct"
export SOLAR_MODEL="upstage/Solar-Open-100B"
export SOLAR_TOKENIZER="upstage/Solar-Open-100B"

export WORK_ROOT="runs/solar_open_100b_opencode_2step"
export BUCKET_DIR="$WORK_ROOT/opencodeinstruct_solar_buckets"
export SPLIT_DIR="$WORK_ROOT/opencodeinstruct_solar_splits"
export PIPELINE_OUT="$WORK_ROOT/pipeline"
export SEED=42
```

## 어떤 파일을 실행하는가

### 1. `generate_trajectory/data/0_bucketing_opencodeinstruct.py`

역할:

- `OpenCodeInstruct`를 읽는다.
- `input -> user`, `output -> assistant`로 보고 Solar chat template 기준 token 길이를 계산한다.
- 전량을 메모리에 올리지 않기 위해 streaming + reservoir sampling을 사용한다.
- 현재는 `--max_samples 40000`을 주어 40K만 먼저 뽑고, 그 40K를 길이순 버킷 파일로 저장한다.

입력:

- 로컬 `jsonl`
- 로컬 `parquet`
- 로컬 디렉터리
- dataset repo id (`nvidia/OpenCodeInstruct`)

출력:

- `bucket_0000_...json`
- `bucket_0001_...json`
- ...

실행:

```bash
uv run python generate_trajectory/data/0_bucketing_opencodeinstruct.py \
  --input_path "$OCI_INPUT_PATH" \
  --input_split train \
  --max_samples 40000 \
  --sample_seed "$SEED" \
  --output_path "$BUCKET_DIR" \
  --tokenizer_path "$SOLAR_TOKENIZER" \
  --chat_template_mode solar \
  --bucket_size 25000 \
  --n_workers 8
```

### 2. `generate_trajectory/data/1_split_opencodeinstruct_2step.py`

역할:

- 버킷 파일들을 읽는다.
- 총 40K를 다시 seed 고정으로 섞는다.
- 정확히 `20K / 20K`로 `split_a`, `split_b`를 만든다.
- 재현용 manifest를 함께 저장한다.

출력:

- `split_a.json`
- `split_b.json`
- `split_manifest.json`

실행:

```bash
uv run python generate_trajectory/data/1_split_opencodeinstruct_2step.py \
  --input_path "$BUCKET_DIR" \
  --output_path "$SPLIT_DIR" \
  --tokenizer_path "$SOLAR_TOKENIZER" \
  --chat_template_mode solar \
  --seed "$SEED" \
  --total_samples 40000
```

### 3. `generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.py`

역할:

- split 파일을 읽는다.
- prompt를 Solar chat template로 감싼다.
- Solar model을 로드하고 Jacobi trajectory를 생성한다.
- 현재는 `torchrun` + `tp_plan="auto"` 기반 `TP4`로 실행된다.
- malformed sample이나 짧은 마지막 block은 skip log에 기록한다.

직접 이 파일을 실행하지는 않고, 아래 shell script가 호출한다.

### 4. `generate_trajectory/generation/solar_modeling_jacobi_forcing_greedy.py`

역할:

- `SolarOpen` 전용 Jacobi forward 구현이다.
- Qwen 전용 내부 API를 쓰지 않고 `SolarOpenModel.forward()`를 직접 호출한다.
- cache, rotary, causal mask는 Solar native 구현에 맡긴다.

이 파일은 Python entry가 내부에서 import해서 사용한다.

### 5. `generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.sh`

역할:

- trajectory 생성 1회를 감싸는 launcher다.
- 현재는 GPU별 shard fan-out이 아니라 `TP4` 단일 실행이다.
- 내부적으로:
  - `CUDA_VISIBLE_DEVICES=4,5,6,7`
  - `uv run torchrun --nproc-per-node 4`
  로 실행한다.
- `MAX_RECORDS`를 주면 split 일부만 예시로 돌릴 수 있다.

### 6. `generate_trajectory/data/tool_merge_single_bucket_data.py`

역할:

- trajectory shard JSON 파일들을 하나의 JSONL로 합친다.

### 7. `generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py`

역할:

- trajectory JSONL을 packed training JSONL로 바꾼다.
- step1에서는 `n=16, w=16`
- step2에서는 `n=32, w=8`

출력:

- `packed_split_a_n16w16.jsonl`
- `packed_split_b_n32w8.jsonl`

### 8. `generate_trajectory/generation/run_solar_opencodeinstruct_2step.sh`

역할:

- 전체 파이프라인 orchestration용 shell script다.
- step1/step2를 순차 실행한다.
- trajectory 생성 -> merge -> packing까지 묶어서 호출한다.

## 실제로 가장 많이 쓰는 명령

### 1. 버킷 생성

```bash
uv run python generate_trajectory/data/0_bucketing_opencodeinstruct.py \
  --input_path "$OCI_INPUT_PATH" \
  --input_split train \
  --max_samples 40000 \
  --sample_seed "$SEED" \
  --output_path "$BUCKET_DIR" \
  --tokenizer_path "$SOLAR_TOKENIZER" \
  --chat_template_mode solar \
  --bucket_size 25000 \
  --n_workers 8
```

### 2. split 생성

```bash
uv run python generate_trajectory/data/1_split_opencodeinstruct_2step.py \
  --input_path "$BUCKET_DIR" \
  --output_path "$SPLIT_DIR" \
  --tokenizer_path "$SOLAR_TOKENIZER" \
  --chat_template_mode solar \
  --seed "$SEED" \
  --total_samples 40000
```

### 3. step1만 예시로 10개 trajectory 생성

이건 디버그/스모크 테스트용이다.

```bash
MODEL_PATH="$SOLAR_MODEL" \
TOKENIZER_PATH="$SOLAR_TOKENIZER" \
SPLIT_DIR="$SPLIT_DIR" \
OUTPUT_ROOT="$PIPELINE_OUT" \
STEP="split_a" \
MAX_RECORDS=10 \
SEED="$SEED" \
GPUS="4 5 6 7" \
CHAT_TEMPLATE_MODE="solar" \
MAX_NEW_SEQ_LEN=2048 \
bash generate_trajectory/generation/run_solar_opencodeinstruct_2step.sh
```

### 4. step1 + step2 전체 순차 실행

`STEP=all`은 step1을 먼저 끝낸 뒤 step2를 이어서 실행한다. 두 step을 동시에 돌리지는 않는다.

```bash
MODEL_PATH="upstage/Solar-Open-100B" \
TOKENIZER_PATH="upstage/Solar-Open-100B" \
SPLIT_DIR="runs/solar_open_100b_opencode_2step/opencodeinstruct_solar_splits" \
OUTPUT_ROOT="runs/solar_open_100b_opencode_2step/pipeline" \
STEP="all" \
SEED="42" \
GPUS="4 5 6 7" \
CHAT_TEMPLATE_MODE="solar" \
MAX_NEW_SEQ_LEN=2048 \
bash generate_trajectory/generation/run_solar_opencodeinstruct_2step.sh
```

#### 4-vLLM. vLLM 경로를 쓸 때

`run_solar_opencodeinstruct_2step.sh`는 현재 HF/torch 기반 경로 기준 설명이다.
vLLM 경로로 step1/step2를 돌릴 때는 `split_a`, `split_b`를 각각 vLLM launcher로
실행해야 한다.

중요:

- vLLM은 독립된 venv에서 실행하는 것을 권장
- 먼저 `/NHNHOME/WORKSPACE/0426030021_A/eunikpark/projects/kllm_workspace/.venv-vllm`를 activate
- 그 다음 `generate_trajectory/generation/generate_trajectory_opencodeinstruct_vllm_greedy.sh`를 사용

예시는 다음과 같다.

step1 (`split_a`):

```bash
cd /NHNHOME/WORKSPACE/0426030021_A/eunikpark/projects/kllm_workspace
source .venv-vllm/bin/activate
cd JacobiForcing-K-LLM

MODEL_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
TOKENIZER_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
STEP="split_a" \
MAX_RECORDS=-1 \
GPUS="4 5 6 7" \
MAX_NEW_SEQ_LEN=2048 \
MAX_NUM_BATCHED_TOKENS=16384 \
MAX_ACTIVE_PROMPTS=10 \
bash generate_trajectory/generation/generate_trajectory_opencodeinstruct_vllm_greedy.sh
```

step2 (`split_b`)도 같은 형식으로 `SPLIT_FILE`과 `SAVE_PATH`만 바꿔서 실행한다.

자세한 설명은
[docs/vllm_trajectory_generation.md](vllm_trajectory_generation.md)에 정리했다.

#### 4-1. step1 전체 실행

```bash
MODEL_PATH="upstage/Solar-Open-100B" \
TOKENIZER_PATH="upstage/Solar-Open-100B" \
SPLIT_DIR="runs/solar_open_100b_opencode_2step/opencodeinstruct_solar_splits" \
OUTPUT_ROOT="runs/solar_open_100b_opencode_2step/pipeline" \
STEP="split_a" \
SEED="42" \
GPUS="4 5 6 7" \
CHAT_TEMPLATE_MODE="solar" \
MAX_NEW_SEQ_LEN=2048 \
bash generate_trajectory/generation/run_solar_opencodeinstruct_2step.sh
```

#### 4-2. step2 전체 실행

```bash
MODEL_PATH="$SOLAR_MODEL" \
TOKENIZER_PATH="$SOLAR_TOKENIZER" \
SPLIT_DIR="$SPLIT_DIR" \
OUTPUT_ROOT="$PIPELINE_OUT" \
STEP="split_b" \
SEED="$SEED" \
GPUS="4 5 6 7" \
CHAT_TEMPLATE_MODE="solar" \
MAX_NEW_SEQ_LEN=2048 \
bash generate_trajectory/generation/run_solar_opencodeinstruct_2step.sh
```

## 실행하면 무엇이 생기는가

### split 결과

- `$SPLIT_DIR/split_a.json`
- `$SPLIT_DIR/split_b.json`
- `$SPLIT_DIR/split_manifest.json`

### step1 결과

- 로그: `$PIPELINE_OUT/step1_split_a_n16w16/logs/tp4.log`
- trajectory shard: `$PIPELINE_OUT/step1_split_a_n16w16/traj_shards/`
- merged trajectory: `$PIPELINE_OUT/step1_split_a_n16w16/merged/trajectory_split_a_n16w16.jsonl`
- packed data: `$PIPELINE_OUT/step1_split_a_n16w16/packed/packed_split_a_n16w16.jsonl`

### step2 결과

- 로그: `$PIPELINE_OUT/step2_split_b_n32w8/logs/tp4.log`
- trajectory shard: `$PIPELINE_OUT/step2_split_b_n32w8/traj_shards/`
- merged trajectory: `$PIPELINE_OUT/step2_split_b_n32w8/merged/trajectory_split_b_n32w8.jsonl`
- packed data: `$PIPELINE_OUT/step2_split_b_n32w8/packed/packed_split_b_n32w8.jsonl`

## 로그를 어떻게 보는가

step1 예시:

```bash
tail -f "$PIPELINE_OUT/step1_split_a_n16w16/logs/tp4.log"
```

shell script 자체도 현재는 `echo`를 추가해 두었기 때문에,

- 어떤 split을 돌리는지
- trajectory generation이 끝났는지
- merge가 시작/완료됐는지
- packing이 시작/완료됐는지

를 터미널에서 바로 볼 수 있다.

## Trajectory Inspection Tool

생성된 trajectory를 detokenize해서 사람이 읽기 쉽게 확인하는 도구는
[tools/inspect_trajectory_decode.md](../tools/inspect_trajectory_decode.md)에 정리돼 있다.

## vLLM FP8 trajectory 생성 경로

느린 HF/torch teacher forward 대신 `nota-ai/Solar-Open-100B-Nota-FP8`를
vLLM offline inference로 띄워 trajectory를 생성하는 경로를 유지한다.

현재 기준:

- 물리 GPU `4,5,6,7`만 사용
- `TP4 + expert parallel enabled`
- 기본 scoring mode는 `apc_multi_prefix`만 사용
- vLLM은 반드시 독립된 venv에서 실행하는 것을 권장

권장 환경:

```bash
cd /NHNHOME/WORKSPACE/0426030021_A/eunikpark/projects/kllm_workspace
source .venv-vllm/bin/activate
cd JacobiForcing-K-LLM
```

현재 vLLM 경로의 자세한 설명은
[docs/vllm_trajectory_generation.md](vllm_trajectory_generation.md)에 정리했다.

### 1. vLLM으로 10개 trajectory 생성

```bash
cd /NHNHOME/WORKSPACE/0426030021_A/eunikpark/projects/kllm_workspace
source .venv-vllm/bin/activate
cd JacobiForcing-K-LLM

MODEL_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
TOKENIZER_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
STEP="split_a" \
MAX_RECORDS=10 \
GPUS="4 5 6 7" \
MAX_NEW_SEQ_LEN=2048 \
MAX_NUM_BATCHED_TOKENS=16384 \
MAX_ACTIVE_PROMPTS=4 \
bash generate_trajectory/generation/generate_trajectory_opencodeinstruct_vllm_greedy.sh
```

### 2. vLLM으로 split_a 전체 생성

```bash
cd /NHNHOME/WORKSPACE/0426030021_A/eunikpark/projects/kllm_workspace
source .venv-vllm/bin/activate
cd JacobiForcing-K-LLM

MODEL_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
TOKENIZER_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
STEP="split_a" \
MAX_RECORDS=-1 \
GPUS="4 5 6 7" \
MAX_NEW_SEQ_LEN=2048 \
MAX_NUM_BATCHED_TOKENS=16384 \
MAX_ACTIVE_PROMPTS=10 \
bash generate_trajectory/generation/generate_trajectory_opencodeinstruct_vllm_greedy.sh
```

### 3. vLLM으로 split_b 전체 생성

```bash
cd /NHNHOME/WORKSPACE/0426030021_A/eunikpark/projects/kllm_workspace
source .venv-vllm/bin/activate
cd JacobiForcing-K-LLM

MODEL_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
TOKENIZER_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
STEP="split_b" \
MAX_RECORDS=-1 \
GPUS="4 5 6 7" \
MAX_NEW_SEQ_LEN=2048 \
MAX_NUM_BATCHED_TOKENS=16384 \
MAX_ACTIVE_PROMPTS=10 \
bash generate_trajectory/generation/generate_trajectory_opencodeinstruct_vllm_greedy.sh
```

현재 validated benchmark 기준:

- `STEP=split_a`
- `1280` records
- `0` skips
