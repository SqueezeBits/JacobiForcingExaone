# Train Full FT with NeMo

## 목적

K-EXAONE 236B MoE full fine-tuning 경로를 문서화한다.

## 메모

- 1 node / 4x B200에서는 viability check 중심으로 본다.
- 메모리, optimizer state, expert parallel 구성 제약을 반드시 고려한다.

## launcher

- `JacobiForcing/scripts/train/train_k_exaone_bridge_full.sh`

