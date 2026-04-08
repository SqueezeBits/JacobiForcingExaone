# K-EXAONE Bridge Design

## 목표

`Megatron-Bridge` PR #2532의 EXAONE4 bridge/provider 패턴을 참고해, K-EXAONE용 별도 bridge/provider를 설계한다.

## 구현 단위

- `integrations/megatron_bridge/models/k_exaone/k_exaone_provider.py`
- `integrations/megatron_bridge/models/k_exaone/k_exaone_bridge.py`
- `integrations/megatron_bridge/models/k_exaone/__init__.py`

## 설계 원칙

- EXAONE4와 같은 family라도 alias 처리하지 않는다.
- K-EXAONE는 별도 model family로 취급한다.
- attention, rope, norm, MoE router/expert 구조는 모두 명시적으로 다시 선언한다.

