# Train LoRA with Megatron Bridge

## 권장 경로

K-EXAONE 236B MoE에 대해서는 LoRA를 기본 경로로 둔다.

## 기본값

- target modules: attention qkv/proj + MLP projection
- rank: 32
- alpha: 64
- dropout: 0.05

## launcher

- `JacobiForcing/scripts/train/train_k_exaone_bridge_lora.sh`

