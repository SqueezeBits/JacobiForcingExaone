#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-4.0-1.2B}"
ATTN_IMPL="${ATTN_IMPL:-flex_attention}"
DTYPE="${DTYPE:-auto}"
DEVICE="${DEVICE:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/smoke_test_exaone4.py" \
  --model-id "${MODEL_ID}" \
  --attn-implementation "${ATTN_IMPL}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  "$@"
