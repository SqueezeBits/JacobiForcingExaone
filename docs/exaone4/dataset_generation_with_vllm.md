# Dataset Generation with vLLM

## 목적

K-EXAONE 메인 데이터 생성 경로는 HF/DeepSpeed가 아니라 `vLLM` batched offline generation으로 둔다.

## 구현

- `JacobiForcing/scripts/data/generate_k_exaone_vllm_dataset.py`
- `JacobiForcing/scripts/data/build_k_exaone_vllm_trainset.sh`

## 권장 버전

- `vllm==0.19.0`
- `transformers 5.5.x` 안정 버전

## 메모

- teacher output은 batched `vLLM` generation으로 만든다.
- intermediate state는 원본 Jacobi trajectory를 최대한 흉내 내는 progressive refinement 후처리로 합성한다.
