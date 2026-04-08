#!/usr/bin/env python3
"""
Smoke test for EXAONE 4.0 models before starting a Jacobi Forcing port.

What this checks:
1. Config / tokenizer / model load
2. Chat template rendering
3. Forward pass with and without KV cache
4. Tiny generation test
5. Optional tiny backward pass to sanity-check training compatibility

This script is intentionally conservative: it aims to fail fast with useful
debug output rather than hide incompatibilities behind silent fallbacks.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = (
    "Write a short Python function named fib(n) that returns the nth Fibonacci "
    "number. Return code only."
)


@dataclass
class CheckResult:
    ok: bool
    detail: str
    extra: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test EXAONE4 compatibility.")
    parser.add_argument(
        "--model-id",
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="HF repo id or local model path.",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Optional tokenizer path. Defaults to --model-id.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flex_attention",
        choices=["flex_attention", "flash_attention_2", "sdpa", "eager"],
        help="Attention implementation to request from transformers.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype for loading the model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens for the generation sanity check.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User prompt used for chat template / generation checks.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True when the tokenizer supports it.",
    )
    parser.add_argument(
        "--run-backward-check",
        action="store_true",
        help="Run a tiny training-style backward pass.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forwarded to Auto* loaders if you are testing a custom checkpoint.",
    )
    return parser.parse_args()


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(dtype_flag: str, device: torch.device) -> torch.dtype:
    if dtype_flag == "float32":
        return torch.float32
    if dtype_flag == "float16":
        return torch.float16
    if dtype_flag == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def maybe_print_cuda_banner(device: torch.device) -> None:
    print(f"[env] torch={torch.__version__}")
    print(f"[env] cuda_available={torch.cuda.is_available()}")
    print(f"[env] selected_device={device}")
    if device.type == "cuda":
        print(f"[env] cuda_device_name={torch.cuda.get_device_name(device)}")


def tokenizer_supports_enable_thinking(tokenizer) -> bool:
    fn = getattr(tokenizer, "apply_chat_template", None)
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "enable_thinking" in sig.parameters


def render_prompt(tokenizer, user_prompt: str, enable_thinking: bool) -> tuple[str, dict[str, Any]]:
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": user_prompt},
    ]

    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    if tokenizer_supports_enable_thinking(tokenizer):
        template_kwargs["enable_thinking"] = enable_thinking

    rendered = tokenizer.apply_chat_template(messages, **template_kwargs)
    return rendered, template_kwargs


def move_inputs_to_device(model_inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in model_inputs.items()}


def summarize_config(config) -> dict[str, Any]:
    rope = getattr(config, "rope_parameters", None)
    if isinstance(rope, dict):
        rope_summary = rope
    else:
        rope_summary = str(rope)

    return {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": getattr(config, "head_dim", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
        "sliding_window": getattr(config, "sliding_window", None),
        "rope_parameters": rope_summary,
    }


def truncate_text(text: str, max_chars: int = 220) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def run_forward_no_cache(model, model_inputs: dict[str, torch.Tensor]) -> CheckResult:
    try:
        with torch.inference_mode():
            outputs = model(**model_inputs, use_cache=False)
        logits = outputs.logits
        return CheckResult(
            ok=True,
            detail="Forward pass without cache succeeded.",
            extra={"logits_shape": list(logits.shape)},
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return CheckResult(
            ok=False,
            detail=f"Forward pass without cache failed: {exc}",
            extra={"traceback": traceback.format_exc()},
        )


def run_forward_with_cache(model, model_inputs: dict[str, torch.Tensor]) -> CheckResult:
    try:
        with torch.inference_mode():
            outputs = model(**model_inputs, use_cache=True)
        pkv = outputs.past_key_values
        seq_len = None
        if hasattr(pkv, "get_seq_length"):
            try:
                seq_len = pkv.get_seq_length()
            except Exception:
                seq_len = None
        return CheckResult(
            ok=True,
            detail="Forward pass with cache succeeded.",
            extra={
                "past_key_values_type": type(pkv).__name__ if pkv is not None else None,
                "cached_seq_length": seq_len,
            },
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return CheckResult(
            ok=False,
            detail=f"Forward pass with cache failed: {exc}",
            extra={"traceback": traceback.format_exc()},
        )


def run_generation(
    model,
    tokenizer,
    model_inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> CheckResult:
    try:
        gen_kwargs = dict(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        if tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**gen_kwargs)
        elapsed = time.perf_counter() - t0

        prompt_len = model_inputs["input_ids"].shape[1]
        new_ids = output_ids[0, prompt_len:]
        decoded = tokenizer.decode(new_ids, skip_special_tokens=False)
        return CheckResult(
            ok=True,
            detail="Generation succeeded.",
            extra={
                "new_tokens": int(new_ids.shape[0]),
                "elapsed_sec": round(elapsed, 4),
                "decoded_preview": truncate_text(decoded),
            },
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return CheckResult(
            ok=False,
            detail=f"Generation failed: {exc}",
            extra={"traceback": traceback.format_exc()},
        )


def run_backward_check(model, model_inputs: dict[str, torch.Tensor]) -> CheckResult:
    try:
        model.train()
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

        labels = model_inputs["input_ids"].clone()
        outputs = model(**model_inputs, labels=labels, use_cache=False)
        loss = outputs.loss
        loss.backward()

        grad_norm = None
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = float(param.grad.detach().float().norm().item())
                break

        model.zero_grad(set_to_none=True)
        model.eval()
        return CheckResult(
            ok=True,
            detail="Backward pass succeeded.",
            extra={
                "loss": float(loss.detach().float().item()),
                "first_grad_norm": grad_norm,
            },
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return CheckResult(
            ok=False,
            detail=f"Backward pass failed: {exc}",
            extra={"traceback": traceback.format_exc()},
        )


def print_result(name: str, result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"[{status}] {name}: {result.detail}")
    if result.extra:
        print(json.dumps(result.extra, indent=2, ensure_ascii=False, default=str))


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer_id = args.tokenizer_id or args.model_id

    maybe_print_cuda_banner(device)
    print(f"[load] model_id={args.model_id}")
    print(f"[load] tokenizer_id={tokenizer_id}")
    print(f"[load] attn_implementation={args.attn_implementation}")
    print(f"[load] dtype={dtype}")

    try:
        config = AutoConfig.from_pretrained(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
        )
        print("[config]")
        print(json.dumps(summarize_config(config), indent=2, ensure_ascii=False, default=str))
    except Exception:
        print("[FAIL] Config load failed.")
        print(traceback.format_exc())
        return 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        print(
            "[tokenizer] "
            + json.dumps(
                {
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "bos_token_id": tokenizer.bos_token_id,
                    "supports_enable_thinking": tokenizer_supports_enable_thinking(tokenizer),
                },
                ensure_ascii=False,
            )
        )
    except Exception:
        print("[FAIL] Tokenizer load failed.")
        print(traceback.format_exc())
        return 1

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            attn_implementation=args.attn_implementation,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
    except Exception:
        print("[FAIL] Model load failed.")
        print(traceback.format_exc())
        return 1

    try:
        rendered_prompt, template_kwargs = render_prompt(
            tokenizer,
            user_prompt=args.prompt,
            enable_thinking=args.enable_thinking,
        )
        print("[chat_template]")
        print(json.dumps(template_kwargs, ensure_ascii=False))
        print(truncate_text(rendered_prompt, max_chars=500))
        model_inputs = tokenizer(rendered_prompt, return_tensors="pt")
        model_inputs = move_inputs_to_device(model_inputs, device)
        print(
            "[input] "
            + json.dumps(
                {
                    "input_ids_shape": list(model_inputs["input_ids"].shape),
                    "attention_mask_shape": list(model_inputs["attention_mask"].shape),
                }
            )
        )
    except Exception:
        print("[FAIL] Chat template / tokenization failed.")
        print(traceback.format_exc())
        return 1

    results: list[tuple[str, CheckResult]] = []
    results.append(("forward_no_cache", run_forward_no_cache(model, model_inputs)))
    results.append(("forward_with_cache", run_forward_with_cache(model, model_inputs)))
    results.append(("generation", run_generation(model, tokenizer, model_inputs, args.max_new_tokens)))

    if args.run_backward_check:
        results.append(("backward", run_backward_check(model, model_inputs)))

    any_fail = False
    for name, result in results:
        print_result(name, result)
        any_fail = any_fail or (not result.ok)

    if device.type == "cuda":
        print(
            "[cuda_memory] "
            + json.dumps(
                {
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                }
            )
        )

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
