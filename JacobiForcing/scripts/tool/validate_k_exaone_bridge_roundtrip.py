#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from integrations.megatron_bridge.models.k_exaone import KExaoneBridge


def main() -> int:
    hf_config = {
        "model_type": "k_exaone",
        "hidden_size": 8192,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 28672,
        "max_position_embeddings": 131072,
        "vocab_size": 131072,
        "rope_theta": 1000000.0,
        "rope_scaling": {"rope_type": "llama3", "factor": 16.0},
        "tie_word_embeddings": True,
        "moe_num_experts": 128,
        "moe_top_k": 8,
        "router_aux_loss_coef": 0.01,
    }

    provider = KExaoneBridge.hf_config_to_provider(hf_config)
    roundtrip = KExaoneBridge.provider_to_hf_config(provider)

    required_keys = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "max_position_embeddings",
        "vocab_size",
        "rope_theta",
        "tie_word_embeddings",
        "moe_num_experts",
        "moe_top_k",
        "router_aux_loss_coef",
    ]
    mismatches = {
        key: {"expected": hf_config.get(key), "actual": roundtrip.get(key)}
        for key in required_keys
        if hf_config.get(key) != roundtrip.get(key)
    }

    print(
        json.dumps(
            {
                "provider_spec": provider.spec.__dict__,
                "roundtrip_config": roundtrip,
                "mismatches": mismatches,
                "ok": len(mismatches) == 0,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
