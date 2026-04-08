from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KExaoneModelSpec:
    model_type: str = "k_exaone"
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    intermediate_size: int = 0
    max_position_embeddings: int = 0
    vocab_size: int = 0
    rope_theta: float = 1000000.0
    rope_scaling: dict = field(default_factory=dict)
    norm_type: str = "rmsnorm"
    attention_variant: str = "gqa"
    moe_num_experts: int = 0
    moe_top_k: int = 0
    moe_router_aux_loss_coef: float = 0.0
    tie_word_embeddings: bool = True

    def lora_target_modules(self) -> list[str]:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]


class KExaoneProvider:
    """Reference provider scaffold for upstreaming K-EXAONE to Megatron Bridge."""

    def __init__(self, spec: KExaoneModelSpec):
        self.spec = spec

    @classmethod
    def from_hf_config(cls, hf_config: dict) -> "KExaoneProvider":
        return cls(
            KExaoneModelSpec(
                hidden_size=hf_config.get("hidden_size", 0),
                num_layers=hf_config.get("num_hidden_layers", 0),
                num_attention_heads=hf_config.get("num_attention_heads", 0),
                num_key_value_heads=hf_config.get("num_key_value_heads", 0),
                intermediate_size=hf_config.get("intermediate_size", 0),
                max_position_embeddings=hf_config.get("max_position_embeddings", 0),
                vocab_size=hf_config.get("vocab_size", 0),
                rope_theta=hf_config.get("rope_theta", 1000000.0),
                rope_scaling=hf_config.get("rope_scaling", {}) or hf_config.get("rope_parameters", {}),
                norm_type=hf_config.get("norm_type", "rmsnorm"),
                attention_variant=hf_config.get("attention_variant", "gqa"),
                moe_num_experts=hf_config.get("num_local_experts", hf_config.get("moe_num_experts", 0)),
                moe_top_k=hf_config.get("moe_top_k", hf_config.get("num_experts_per_tok", 0)),
                moe_router_aux_loss_coef=hf_config.get("router_aux_loss_coef", 0.0),
                tie_word_embeddings=hf_config.get("tie_word_embeddings", True),
            )
        )

    def to_megatron_config(self) -> dict:
        return {
            "num_layers": self.spec.num_layers,
            "hidden_size": self.spec.hidden_size,
            "num_attention_heads": self.spec.num_attention_heads,
            "num_query_groups": self.spec.num_key_value_heads,
            "ffn_hidden_size": self.spec.intermediate_size,
            "seq_length": self.spec.max_position_embeddings,
            "vocab_size": self.spec.vocab_size,
            "rotary_base": self.spec.rope_theta,
            "rotary_scaling": self.spec.rope_scaling,
            "normalization": self.spec.norm_type,
            "moe_router_topk": self.spec.moe_top_k,
            "num_moe_experts": self.spec.moe_num_experts,
            "moe_aux_loss_coeff": self.spec.moe_router_aux_loss_coef,
            "share_embeddings_and_output_weights": self.spec.tie_word_embeddings,
        }
