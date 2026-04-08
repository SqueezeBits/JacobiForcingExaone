# EXAONE4 Additions Summary

## 목적

이 문서는 `JacobiForcing` 저장소에 대해 `EXAONE-4.0-1.2B` 지원을 위해 추가/수정한 내용만 따로 정리한 요약 문서다.

범위:

- EXAONE4 smoke test
- EXAONE4 greedy Jacobi trajectory 초안
- EXAONE4 trajectory -> packed data 검증
- OpenCodeInstruct용 EXAONE4 pipeline script
- 관련 bug fix

생성물(`tmp/`, `__pycache__/`)은 제외하고, 실제 코드/스크립트 변경만 적었다.

---

## 1. 새로 추가한 파일

## 1.1 문서

### [EXAONE4_JACOBI_PORT_PLAN.md](/workspace/exaone_workspace/JacobiForcing/EXAONE4_JACOBI_PORT_PLAN.md)

- EXAONE4 포팅 계획서
- 단계별 작업 순서, 검증 포인트, 파일 단위 TODO 정리

### [JACOBI_FORCING_REPO_ANALYSIS.md](/workspace/exaone_workspace/JacobiForcing/JACOBI_FORCING_REPO_ANALYSIS.md)

- 저장소 전체 분석 문서
- 논문/블로그/코드 일치 여부까지 포함한 구조 설명

---

## 1.2 EXAONE4 trajectory / 모델 포팅

### [exaone4_modeling_jacobi_forcing_greedy.py](/workspace/exaone_workspace/JacobiForcing/generate_trajectory/generation/exaone4_modeling_jacobi_forcing_greedy.py)

- EXAONE4용 greedy Jacobi trajectory 커널 초안
- EXAONE4 HF forward를 직접 사용
- `DynamicCache.crop(...)` 기반 cache trim 사용
- 모든 `answer_trajectory_ids` step을 `n_token_seq_len` 고정 길이로 맞춤

### [generate_trajectory_exaone4_greedy.py](/workspace/exaone_workspace/JacobiForcing/generate_trajectory/generation/generate_trajectory_exaone4_greedy.py)

- EXAONE4용 trajectory generation 엔트리
- 입력:
  - JSON
  - JSONL
  - `input` / `prompt` / `text` 필드 지원
- 출력:
  - `diffusion_itr_id`
  - `data_id`
  - `prompt_ids`
  - `answer_trajectory_ids`
  - `teacher_output_ids`
- 최근에 `batch_size` 지원 추가

### [generate_trajectory_exaone4_greedy.sh](/workspace/exaone_workspace/JacobiForcing/generate_trajectory/generation/generate_trajectory_exaone4_greedy.sh)

- README 스타일의 EXAONE4 trajectory wrapper
- workspace 기본 경로 사용
- GPU 수 자동 감지
- `BATCH_SIZE` 환경변수 지원

---

## 1.3 EXAONE4 검증 스크립트

### [smoke_test_exaone4.py](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/smoke_test_exaone4.py)

- EXAONE4 config / tokenizer / model load 테스트
- chat template
- forward no-cache / with-cache
- generation
- optional backward

### [run_smoke_test_exaone4.sh](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/run_smoke_test_exaone4.sh)

- smoke test 실행 wrapper

### [validate_exaone4_jacobi_greedy.py](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/validate_exaone4_jacobi_greedy.py)

- EXAONE4 greedy Jacobi block 결과가 AR greedy prefix와 일치하는지 검증

### [validate_exaone4_training_step.py](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/validate_exaone4_training_step.py)

- packed sample을 EXAONE4에 넣어 Jacobi Forcing 스타일 loss 계산 확인
- optional backward로 grad 생성 확인

---

## 1.4 OpenCodeInstruct / tiny dataset 보조 스크립트

### [download_opencodeinstruct_to_jsonl.py](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/download_opencodeinstruct_to_jsonl.py)

- `nvidia/OpenCodeInstruct`를 HF에서 받아 JSONL shard로 저장

### [sample_prompts_small.json](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/sample_prompts_small.json)

- 작은 EXAONE4 trajectory / training smoke test용 샘플 prompt 세트

### [build_tiny_exaone4_trainset.sh](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/build_tiny_exaone4_trainset.sh)

- tiny prompt 세트 -> trajectory -> packed data까지 end-to-end

### [build_opencodeinstruct_exaone4_train_pipeline.sh](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/build_opencodeinstruct_exaone4_train_pipeline.sh)

- OpenCodeInstruct JSONL dir -> bucket -> trajectory -> packed data end-to-end
- `START_STAGE` resume 지원
- `OVERWRITE_PACKED` 지원
- GPU 수 자동 감지
- `BATCH_SIZE` 지원

### [run_generate_exaone4_trajectory.sh](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/run_generate_exaone4_trajectory.sh)

- 단일 입력 파일 기준 trajectory generation wrapper

---

## 2. 수정한 기존 파일

## [0_bucketing_opencodeinstruct.py](/workspace/exaone_workspace/JacobiForcing/generate_trajectory/data/0_bucketing_opencodeinstruct.py)

수정 이유:

- EXAONE tokenizer와 multiprocessing worker 초기화가 바로 동작하지 않았음
- `apply_chat_template(..., return_tensors="pt")` 결과가 `BatchEncoding`인 경우를 처리하지 못했음

수정 내용:

- worker `init_worker()`에 `tokenizer_path`를 `initargs`로 전달
- `BatchEncoding` / tensor / list 형태를 모두 처리하도록 tokenization output 파싱 수정
- 에러 출력 개선

효과:

- EXAONE tokenizer로 OpenCodeInstruct bucketting 가능

---

## 3. 현재 검증 완료 상태

## 3.1 EXAONE4 환경/모델 검증

완료:

- EXAONE4 config 로드
- EXAONE4 tokenizer 로드
- `flex_attention` 로드
- forward without cache
- forward with cache
- generation

확인 결과:

- `model_type=exaone4`
- `num_hidden_layers=30`
- `hidden_size=2048`
- `num_attention_heads=32`
- `num_key_value_heads=8`

---

## 3.2 EXAONE4 greedy Jacobi correctness

완료:

- `block_size=8`에서 AR greedy prefix와 exact match
- `block_size=16`에서 AR greedy prefix와 exact match

핵심 수정:

- generation phase에서 `attention_mask=None`을 써야 EXAONE4 cache continuation이 AR와 일치함

---

## 3.3 Packed data / training-step 검증

완료:

- EXAONE4 trajectory JSON 생성
- packed JSONL 변환 성공
- EXAONE4에 packed sample을 넣어 loss 계산 성공
- backward 성공

즉, 현재는:

```text
EXAONE4
  -> trajectory generation
  -> packed training data
  -> single training step
```

까지는 end-to-end 검증됨

---

## 3.4 Tiny / batched dataset 검증

완료:

- tiny sample prompt 세트로 trajectory 생성 성공
- packed 데이터 생성 성공
- `generate_trajectory_exaone4_greedy.py`에 `batch_size` 경로 추가
- `batch_size=2` 샘플 검증 성공
- batched trajectory도 packed 변환 통과

---

## 4. 현재 설계의 한계

### 1. EXAONE4 trajectory batch support는 1차 초안

현재 배치 경로는:

- cache-aware batched Jacobi kernel이 아니라
- stateless batched full-sequence refinement 방식

장점:

- 구현 단순
- batch support 빠르게 확보
- util 개선의 첫 발판

단점:

- cache reuse가 적음
- 최종 고성능 trajectory collector는 아님

### 2. inference_engine 포팅은 아직 안 함

현재 EXAONE4 포팅은 HF 경로 기준이다.

아직 안 한 것:

- `inference_engine/models/exaone4.py`
- EXAONE4 inference backend
- cache-aware high-performance serving path

### 3. multiblock MR 포팅은 아직 안 함

현재는 vanilla greedy Jacobi trajectory 중심이다.

아직 안 한 것:

- EXAONE4용 multiblock decoding
- rejection recycling

---

## 5. 지금 바로 쓰는 진입점

### OpenCodeInstruct JSONL 받기

- [download_opencodeinstruct_to_jsonl.py](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/download_opencodeinstruct_to_jsonl.py)

### OpenCodeInstruct -> bucket -> trajectory -> packed

- [build_opencodeinstruct_exaone4_train_pipeline.sh](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/build_opencodeinstruct_exaone4_train_pipeline.sh)

### EXAONE4 trajectory만 단일 파일 기준으로 생성

- [generate_trajectory_exaone4_greedy.sh](/workspace/exaone_workspace/JacobiForcing/generate_trajectory/generation/generate_trajectory_exaone4_greedy.sh)

### 작은 샘플 end-to-end

- [build_tiny_exaone4_trainset.sh](/workspace/exaone_workspace/JacobiForcing/JacobiForcing/scripts/exaone4/build_tiny_exaone4_trainset.sh)

---

## 6. 추천 다음 단계

우선순위 기준:

1. EXAONE4 packed dataset으로 tiny pilot training launcher 만들기
2. batched trajectory generation 성능 프로파일링
3. cache-aware batched Jacobi trajectory collector로 2차 최적화
4. EXAONE4 multiblock MR 포팅
5. inference_engine 포팅
