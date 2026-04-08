# K-EXAONE on Megatron Bridge / NeMo

## 목적

이 문서는 K-EXAONE 236B MoE를 이 저장소의 메인 학습 타깃으로 삼을 때, 왜 `Megatron Bridge / NeMo`를 기준 스택으로 선택하는지 정리한다.

## 권장 스택

- 데이터 생성: `vLLM`
- LoRA 학습: `Megatron Bridge PEFT`
- Full fine-tuning: `NeMo / Megatron Bridge`
- 환경 관리: `pixi`

## 왜 이 스택인가

- `236B MoE`는 expert parallel, tensor parallel, pipeline parallel을 모두 고려해야 한다.
- 현재 HF/DeepSpeed 경로는 작은 모델 검증에는 적합하지만, K-EXAONE 236B MoE를 메인 학습 경로로 삼기에는 한계가 크다.
- Megatron Bridge는 HF 체크포인트 import/export와 Megatron 학습 스택 사이를 연결하는 구조를 제공한다.
- NeMo는 launcher, recipe, 운영 관점에서 상위 레이어 역할을 한다.

