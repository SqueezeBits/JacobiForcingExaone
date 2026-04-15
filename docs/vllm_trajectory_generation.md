# vLLM Trajectory Generation

이 문서는 `nota-ai/Solar-Open-100B-Nota-FP8`를 vLLM으로 띄워
JacobiForcing trajectory를 생성하는 현재 기준 경로를 정리한다.

기준 경로는 `JacobiForcing-K-LLM` 리포 루트다.

## 핵심 결정

- 모델: `nota-ai/Solar-Open-100B-Nota-FP8`
- 토크나이저: `nota-ai/Solar-Open-100B-Nota-FP8`
- GPU: 물리 GPU `4,5,6,7`만 사용
- 병렬화: `TP4 + expert parallel enabled`
- block 설정: 기본 예시는 `n=16`
- 현재 기본 scoring mode: `apc_multi_prefix`
- 첫 토큰은 `prefill_greedy`
- block의 나머지 greedy token은 APC 기반 scoring으로 계산

## 환경

vLLM은 반드시 독립된 venv에서 실행하는 것을 권장한다.

이 리포 기준 권장 환경:

```bash
cd /NHNHOME/WORKSPACE/0426030021_A/eunikpark/projects/kllm_workspace
source .venv-vllm/bin/activate
```

이유:

- 일반 학습/유틸 환경과 vLLM 의존성이 다르다
- `torch`, `transformers`, `flashinfer`, CUDA 관련 패키지 충돌을 줄인다
- trajectory shell script가 plain `import vllm`에 의존하므로, 활성화된 venv에서
  `vllm` import가 바로 되어야 한다

## 현재 실행 경로

엔트리 포인트:

- `generate_trajectory/generation/generate_trajectory_opencodeinstruct_vllm_greedy.sh`
- `generate_trajectory/generation/generate_trajectory_opencodeinstruct_vllm_greedy.py`

현재 shell script는 `STEP`에 따라 split과 block size를 자동 선택한다.

## 왜 `apc_multi_prefix`를 기본으로 쓰는가

`10`개 prompt benchmark에서 `apc_multi_prefix`가 `1280` records, `0` skips로
valid했고, 현재 production 기본값으로 사용한다.

## 권장 실행 방법

### 1. vLLM 전용 venv 활성화

```bash
cd /NHNHOME/WORKSPACE/0426030021_A/eunikpark/projects/kllm_workspace
source .venv-vllm/bin/activate
cd JacobiForcing-K-LLM
```

### 2. 10개 prompt benchmark

```bash
MODEL_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
TOKENIZER_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
STEP="split_a" \
MAX_RECORDS=10 \
GPUS="4 5 6 7" \
MAX_NEW_SEQ_LEN=2048 \
MAX_NUM_BATCHED_TOKENS=16384 \
MAX_ACTIVE_PROMPTS=10 \
bash generate_trajectory/generation/generate_trajectory_opencodeinstruct_vllm_greedy.sh
```

### 3. split_a 전체 실행

```bash
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

### 4. split_b 전체 실행

```bash
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

### 5. 시간까지 보고 싶을 때

```bash
time MODEL_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
TOKENIZER_PATH="nota-ai/Solar-Open-100B-Nota-FP8" \
STEP="split_a" \
MAX_RECORDS=10 \
GPUS="4 5 6 7" \
MAX_NEW_SEQ_LEN=2048 \
MAX_NUM_BATCHED_TOKENS=16384 \
MAX_ACTIVE_PROMPTS=10 \
bash generate_trajectory/generation/generate_trajectory_opencodeinstruct_vllm_greedy.sh
```

## 주요 파라미터

- `STEP`
  - `split_a` 또는 `split_b`
  - `split_a`면 `n=16`, `split_b`면 `n=32`
- `MAX_NUM_BATCHED_TOKENS`
  - 현재 기본값 `16384`
- `MAX_ACTIVE_PROMPTS`
  - batched multi-prompt scheduler에서 동시에 전진시키는 prompt 수
- `MAX_RECORDS`
  - `10`이면 benchmark, `-1`이면 split 전체 실행

## 출력

기본 출력 파일:

- trajectory:
  - `runs/solar_open_100b_opencode_2step/pipeline_vllm/step1_split_a_n16w16/traj_shards/split_a_solar_solar_vllm_greedy_jacobi_len16_0_10.json`
- skipped:
  - `runs/solar_open_100b_opencode_2step/pipeline_vllm/step1_split_a_n16w16/traj_shards/split_a_solar_solar_vllm_greedy_jacobi_len16_0_10_skipped.jsonl`
- stats:
  - `runs/solar_open_100b_opencode_2step/pipeline_vllm/step1_split_a_n16w16/traj_shards/split_a_solar_solar_vllm_greedy_jacobi_len16_0_10_stats.json`
- runtime log:
  - `runs/solar_open_100b_opencode_2step/pipeline_vllm/step1_split_a_n16w16/traj_shards/logs/vllm_tp4_ep4.log`

## 검증 기준

현재 valid run 기준 체크 항목:

- `1280` records
- `0` skipped records
- prompt `10`개 모두 완료
- prompt당 `128` iterations
- final block length `16`

## 참고

- vLLM logger의 `Running: 0 reqs`, `GPU KV cache usage: 0.0%`는 offline bursty
  workload에서 마지막 snapshot만 찍는 방식 때문에 misleading할 수 있다
- 로컬 vLLM logger는 interval max를 같이 찍도록 패치되어 있다
