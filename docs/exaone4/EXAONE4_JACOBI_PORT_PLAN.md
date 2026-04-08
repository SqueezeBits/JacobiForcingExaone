# EXAONE 4.0 -> Jacobi Forcing Port Plan

## 목적

이 문서는 `JacobiForcing` 저장소를 `EXAONE-4.0-1.2B`에 적용하기 위한 실제 작업 순서를 정리한 포팅 계획서다.

대상 모델:

- `LGAI-EXAONE/EXAONE-4.0-1.2B`

핵심 전제:

- 현재 저장소의 **학습 경로는 비교적 모델-agnostic** 하다.
- 반면 **trajectory generation / custom Jacobi decoding / inference engine**은 Qwen 계열 구현에 강하게 묶여 있다.
- 따라서 포팅 우선순위는 다음과 같다.

```text
1) EXAONE4 로딩/마스킹/캐시 호환성 확인
2) EXAONE4용 Jacobi trajectory 생성 커널 구현
3) 기존 data prepare 재사용
4) 기존 train 경로 재사용
5) vanilla Jacobi가 붙으면 multiblock/recycling 확장
6) inference_engine 포팅은 마지막
```

---

## 1. 현재 저장소에서 재사용 가능한 부분 / 새로 만들어야 하는 부분

## 1.1 거의 그대로 재사용 가능한 부분

### 학습 진입점

- [soft_flexattn_train_cllm_multiblock.py](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/train/soft_flexattn_train_cllm_multiblock.py)
- [soft_flexattn_cllm_trainer_multiblock.py](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock.py)

이유:

- `AutoConfig`, `AutoModelForCausalLM`, `AutoTokenizer`를 사용한다
- 학습 로직의 핵심은 모델 구조보다 "packed sequence + custom attention mask + logits indexing"에 의존한다
- EXAONE4는 HF 구현에서 `flex_attention` 지원이 명시되어 있어 학습 경로와 궁합이 좋다

### 데이터 후처리 / packing

- [2_prepare_efficient_cllm_training_data_progressive_noise_window.py](/workspace/exaone_workspace/JacobiForcing/generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py)
- 그 외 `generate_trajectory/data/2_*`, `3_*` 계열

이유:

- 이 단계는 모델 아키텍처보다 입력 JSON 포맷에 의존한다
- 아래 필드만 맞으면 재사용 가능하다
  - `prompt_ids`
  - `answer_trajectory_ids`
  - `teacher_output_ids`

---

## 1.2 새로 만들어야 하는 부분

### EXAONE4용 Jacobi trajectory 생성 커널

현재 Qwen 전용:

- [generate_trajectory_opencodeinstruct_greedy.py](/workspace/exaone_workspace/JacobiForcing/generate_trajectory/generation/generate_trajectory_opencodeinstruct_greedy.py)
- [qwen2_modeling_jacobi_forcing_greedy.py](/workspace/exaone_workspace/JacobiForcing/generate_trajectory/generation/qwen2_modeling_jacobi_forcing_greedy.py)

새로 필요한 것:

- `generate_trajectory/generation/exaone4_modeling_jacobi_forcing_greedy.py`
- `generate_trajectory/generation/generate_trajectory_exaone4_greedy.py`

이유:

- 현재 trajectory 생성은 `Qwen2ForCausalLM` monkey-patch 방식에 직접 결합되어 있다
- EXAONE4는 attention forward, cache update, RoPE 처리, layer wiring이 다르다

### EXAONE4용 multiblock MR 커널

현재 Qwen 전용:

- [cllm2_qwen2_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified.py](/workspace/exaone_workspace/JacobiForcing/modeling/cllm2_qwen2_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified.py)

새로 필요한 것:

- `modeling/exaone4_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified.py`

주의:

- 이 파일은 난도가 높다
- vanilla Jacobi가 먼저 붙기 전에는 바로 들어가면 안 된다

### EXAONE4용 inference engine backend

현재 전용 구현:

- [qwen3.py](/workspace/exaone_workspace/JacobiForcing/inference_engine/models/qwen3.py)

새로 필요한 것:

- `inference_engine/models/exaone4.py`
- 필요 시 `utils/loader.py`, `config.py`, `model_runner.py` 일부 수정

주의:

- 연구 재현 단계에서는 우선순위가 낮다
- HF 경로에서 vanilla/multiblock Jacobi가 먼저 검증되어야 한다

---

## 2. 권장 작업 순서

## Phase 0. 모델 호환성 체크

목표:

- EXAONE4가 현재 학습 경로에서 바로 올라오는지 확인

할 일:

1. `AutoConfig.from_pretrained("LGAI-EXAONE/EXAONE-4.0-1.2B")`
2. `AutoModelForCausalLM.from_pretrained(...)`
3. `AutoTokenizer.from_pretrained(...)`
4. `attn_implementation="flex_attention"`로 로드 가능한지 확인
5. `use_cache=True/False` 모두 정상 동작하는지 확인
6. `apply_chat_template` 출력 형식 확인
7. `pad_token_id`, `eos_token_id`, special token 집합 확인

성공 기준:

- 단일 prompt forward가 된다
- KV cache generation이 된다
- loss 계산 forward가 된다

산출물:

- 간단한 smoke test script

추천 파일:

- `scripts/exaone4/smoke_test_exaone4.py`

---

## Phase 1. EXAONE4용 vanilla Jacobi trajectory 생성

목표:

- EXAONE4에서 `answer_trajectory_ids`를 만들 수 있어야 한다

할 일:

1. Qwen용 greedy Jacobi 커널을 EXAONE4용으로 복제
2. 다음 기능을 동일하게 구현
   - prefill phase
   - generation phase
   - draft 초기화
   - longest matching prefix acceptance
   - rejected tail KV trim
   - EOS 처리
3. `get_jacobi_forward_trajectory_greedy` 또는 이에 준하는 메서드 이름으로 EXAONE4 모델에 붙이기
4. trajectory generation 스크립트를 EXAONE4 tokenizer/chat template에 맞게 교체

추가 결정 포인트:

- `enable_thinking=False`로 시작할지
- reasoning 데이터를 다룰 때 `<think>`를 trajectory에 포함할지

권장 시작값:

- 첫 포팅은 `enable_thinking=False`
- reasoning mode는 나중에 별도 실험

새 파일 제안:

- `generate_trajectory/generation/exaone4_modeling_jacobi_forcing_greedy.py`
- `generate_trajectory/generation/generate_trajectory_exaone4_greedy.py`

성공 기준:

- 한 prompt에서 intermediate trajectory가 여러 step 생성된다
- 최종 `teacher_output_ids`가 vanilla AR 결과와 문맥상 일관된다
- JSON 포맷이 기존 data prepare와 호환된다

---

## Phase 2. 기존 data prepare 파이프라인 재사용

목표:

- EXAONE4 trajectory를 학습용 packed sequence로 변환

할 일:

1. `generate_trajectory_exaone4_greedy.py` 출력 필드를 기존 포맷과 맞춘다
2. 기존 스크립트로 packed JSONL을 만든다

사용 스크립트:

- [2_prepare_efficient_cllm_training_data_progressive_noise_window.py](/workspace/exaone_workspace/JacobiForcing/generate_trajectory/data/2_prepare_efficient_cllm_training_data_progressive_noise_window.py)

검증 포인트:

- `prompt_ids_len`
- `complete_training_sequence_ids`
- `traj_position_indices`
- sequence length 분포
- special token이 과도하게 들어가지 않는지

성공 기준:

- 학습기가 JSONL을 읽고 배치 구성이 된다
- sample 하나를 trainer에 넣었을 때 길이 mismatch가 없다

---

## Phase 3. EXAONE4로 vanilla Jacobi Forcing 학습

목표:

- 기존 trainer를 EXAONE4에 그대로 태워서 loss가 내려가는지 확인

핵심 파일:

- [soft_flexattn_train_cllm_multiblock.py](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/train/soft_flexattn_train_cllm_multiblock.py)
- [soft_flexattn_cllm_trainer_multiblock.py](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock.py)

먼저 확인할 것:

1. EXAONE4 HF 구현이 `flex_attention`을 실제로 지원하는지
2. custom block mask를 `attention_mask` 자리에 넣는 현재 방식이 EXAONE4 forward에서도 허용되는지
3. `position_ids`를 pair-shared 방식으로 넣는 것이 EXAONE4 RoPE 구현과 충돌하지 않는지

권장 첫 실험:

- block size: `n=16` 또는 `n=32`
- `K=1` 개념의 vanilla 데이터로 시작
- 짧은 max sequence로 pilot
- batch size 1 유지

fine-tuning 방식 추천:

- 빠른 검증 1차: `QLoRA` 가능
- 최종 실험: `Full fine-tuning` 권장

이유:

- 이 repo의 기본 재현 경로는 full FT다
- 다만 EXAONE4 1.2B는 비교적 작아서 full FT도 가능성이 높다
- 먼저 작은 pilot을 빠르게 돌리고 싶으면 QLoRA가 효율적이다

성공 기준:

- `loss_ar`, `loss_consistency` 둘 다 NaN 없이 계산된다
- 짧은 학습에서도 draft quality가 baseline 대비 좋아진다

---

## Phase 4. EXAONE4에서 vanilla Jacobi decoding 검증

목표:

- 학습된 EXAONE4가 실제로 Jacobi decoding에서 더 긴 accepted prefix를 보이는지 확인

할 일:

1. EXAONE4용 `jacobi_forward_greedy` 추론 스크립트 작성
2. 동일 prompt에서
   - AR baseline
   - Jacobi decoded EXAONE4
   를 비교
3. 측정 항목 수집
   - average accepted prefix
   - tokens per forward
   - tokens per second
   - quality regression 여부

추천 파일:

- `applications/exaone4_jacobi_chat.py`
- `scripts/exaone4/eval_vanilla_jacobi.py`

성공 기준:

- vanilla Jacobi가 정상 종료
- accepted prefix 평균이 1보다 커짐
- AR 대비 품질이 과도하게 붕괴하지 않음

---

## Phase 5. multiblock + rejection recycling 포팅

목표:

- 논문에서 강조하는 고성능 decoding variant를 EXAONE4에 이식

할 일:

1. Qwen용 multiblock 구현을 EXAONE4용으로 복제
2. 아래 로직을 순차적으로 이식
   - real-active block
   - pseudo-active block
   - `spawn_threshold = ceil(r * n_token_seq_len)`
   - n-gram pool
   - candidate recycling
   - block promotion/switching
3. KV cache batch expansion/trim 로직을 EXAONE4 cache 규약에 맞게 수정

핵심 파일:

- 현재 구현: [cllm2_qwen2_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified.py](/workspace/exaone_workspace/JacobiForcing/modeling/cllm2_qwen2_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified.py)
- 새 구현 후보: `modeling/exaone4_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified.py`

주의:

- 이 단계는 vanilla Jacobi 성공 전에는 들어가면 안 된다
- 가장 어려운 부분은 cache shape와 candidate batch expansion이다

성공 기준:

- `K=2`부터 정상 동작
- recycling이 붙어도 quality regression이 과하지 않음
- TPS / TPF 개선이 관찰됨

---

## Phase 6. inference_engine 포팅

목표:

- 서빙 친화적 EXAONE4 Jacobi backend 확보

현재 상태:

- `inference_engine/`는 Qwen 계열 전용 구현이 많다
- 특히 모델 구현이 [qwen3.py](/workspace/exaone_workspace/JacobiForcing/inference_engine/models/qwen3.py)에 묶여 있다

할 일:

1. EXAONE4 전용 model backend 작성
2. loader가 EXAONE4 HF weights를 읽도록 확장
3. attention / RoPE / QK norm 규약 반영
4. vanilla Jacobi부터 연결
5. 이후 multiblock MR 지원 추가

새 파일 제안:

- `inference_engine/models/exaone4.py`

수정 가능성 있는 파일:

- `inference_engine/utils/loader.py`
- `inference_engine/config.py`
- `inference_engine/engine/model_runner.py`

우선순위:

- 낮음
- 연구 재현이 끝난 뒤 진행

---

## 3. 파일 단위 작업 목록

## 새로 추가할 파일

- `generate_trajectory/generation/exaone4_modeling_jacobi_forcing_greedy.py`
- `generate_trajectory/generation/generate_trajectory_exaone4_greedy.py`
- `scripts/exaone4/smoke_test_exaone4.py`
- `scripts/exaone4/prepare_exaone4_trajectory.sh`
- `scripts/exaone4/train_exaone4_jacobi.sh`
- `scripts/exaone4/eval_vanilla_jacobi.py`
- `applications/exaone4_jacobi_chat.py`

2차 단계 이후:

- `modeling/exaone4_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified.py`
- `inference_engine/models/exaone4.py`

## 수정할 가능성이 큰 파일

- `JacobiForcing/train/soft_flexattn_train_cllm_multiblock.py`
  - EXAONE4에서 필요한 로드 옵션 추가
- `JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock.py`
  - EXAONE4와 `flex_attention` 호환성 문제 발생 시 최소 수정
- `README.md`
  - EXAONE4 사용법 추가

---

## 4. 첫 주차 기준 현실적인 마일스톤

### Milestone A

- EXAONE4 smoke test 통과
- tokenizer/chat template/eos/pad 규약 정리 완료

### Milestone B

- EXAONE4용 trajectory generation 완료
- 기존 `2_prepare_*`로 packed data 생성 성공

### Milestone C

- vanilla Jacobi Forcing 학습 pilot 1회 성공
- loss 하강 및 짧은 qualitative sample 확인

### Milestone D

- vanilla Jacobi inference에서 accepted prefix 개선 확인

그다음에야 multiblock으로 가는 것이 좋다.

---

## 5. 가장 먼저 해야 할 일

지금 바로 시작한다면 순서는 아래가 가장 좋다.

1. `scripts/exaone4/smoke_test_exaone4.py` 작성
2. EXAONE4 tokenizer/chat template 확인
3. `exaone4_modeling_jacobi_forcing_greedy.py` 초안 작성
4. 단일 prompt trajectory 생성 테스트
5. packed data 100개만 만들어 trainer에 넣어보기

즉, **첫 목표는 multiblock이 아니라 "EXAONE4에서 vanilla Jacobi trajectory를 안정적으로 뽑는 것"** 이다.

---

## 6. 추천 전략 요약

가장 추천하는 전략은:

- 모델: `EXAONE-4.0-1.2B`
- mode: `enable_thinking=False`
- 데이터: code task부터 시작
- trajectory: greedy vanilla Jacobi
- 학습: `n=16` 또는 `n=32`, 짧은 pilot
- fine-tuning: 빠른 확인은 QLoRA, 최종은 full FT
- 추론: vanilla Jacobi 성공 후 multiblock MR

이 순서를 따르면 리스크를 가장 잘 통제할 수 있다.
