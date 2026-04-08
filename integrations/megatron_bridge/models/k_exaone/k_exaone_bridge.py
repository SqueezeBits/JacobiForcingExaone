from __future__ import annotations

from .k_exaone_provider import KExaoneProvider


class KExaoneBridge:
    """Reference Hugging Face <-> Megatron Bridge mapping scaffold for K-EXAONE."""

    model_type = "k_exaone"

    @classmethod
    def hf_config_to_provider(cls, hf_config: dict) -> KExaoneProvider:
        return KExaoneProvider.from_hf_config(hf_config)

    @classmethod
    def provider_to_hf_config(cls, provider: KExaoneProvider) -> dict:
        spec = provider.spec
        return {
            "model_type": cls.model_type,
            "hidden_size": spec.hidden_size,
            "num_hidden_layers": spec.num_layers,
            "num_attention_heads": spec.num_attention_heads,
            "num_key_value_heads": spec.num_key_value_heads,
            "intermediate_size": spec.intermediate_size,
            "max_position_embeddings": spec.max_position_embeddings,
            "vocab_size": spec.vocab_size,
            "rope_theta": spec.rope_theta,
            "rope_scaling": spec.rope_scaling,
            "tie_word_embeddings": spec.tie_word_embeddings,
            "moe_num_experts": spec.moe_num_experts,
            "moe_top_k": spec.moe_top_k,
            "router_aux_loss_coef": spec.moe_router_aux_loss_coef,
        }

    @classmethod
    def export_mapping(cls) -> dict:
        return {
            "q_proj": "self_attention.query_projection",
            "k_proj": "self_attention.key_projection",
            "v_proj": "self_attention.value_projection",
            "o_proj": "self_attention.dense",
            "gate_proj": "mlp.gate_proj",
            "up_proj": "mlp.up_proj",
            "down_proj": "mlp.down_proj",
            "router": "mlp.router",
            "experts": "mlp.experts",
        }

