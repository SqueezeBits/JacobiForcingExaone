#!/usr/bin/env python3
import argparse
import gc
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline benchmark for HF baseline, Jacobi Forcing, and optional vLLM baseline."
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "jacobi", "vllm", "nano_vllm_ar", "nano_vllm_jacobi"],
        required=True,
    )
    parser.add_argument("--dataset", type=str, required=True, help="Input dataset: parquet, jsonl, or json.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-generations-jsonl", type=str, default=None)
    parser.add_argument("--text-key", type=str, default="prompt")
    parser.add_argument("--task-id-key", type=str, default="task_id")
    parser.add_argument("--prompt-format", choices=["humaneval", "plain"], default="humaneval")
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0, help="Measured sample count. 0 means all available.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup sample count before measurement.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attention-impl", type=str, default="flash_attention_2")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--alt-eos-id", type=int, default=151645, help="-1 disables the extra EOS id.")
    parser.add_argument("--synchronize-cuda", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--n-token-seq-len", type=int, default=64)
    parser.add_argument("--jacobi-K", type=int, default=2)
    parser.add_argument("--jacobi-r", type=float, default=0.85)
    parser.add_argument("--jacobi-n-gram-pool-size", type=int, default=4)
    parser.add_argument("--jacobi-lookahead-start-ratio", type=float, default=0.0)
    parser.add_argument("--jacobi-max-calls", type=int, default=128)

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--nano-vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    if args.backend == "vllm":
        os_env = __import__("os").environ
        os_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_cuda(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def load_records(dataset_path: str, limit: int) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        records = pd.read_parquet(path).to_dict(orient="records")
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Expected top-level list in JSON dataset: {path}")
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    normalized = []
    for idx, record in enumerate(records):
        if isinstance(record, str):
            normalized.append({"task_id": f"idx_{idx}", "prompt": record})
        else:
            normalized.append(record)
    if limit > 0:
        normalized = normalized[:limit]
    return normalized


def build_prompt(raw_text: str, prompt_format: str) -> str:
    raw_text = raw_text.strip()
    if prompt_format == "plain":
        return raw_text
    return (
        "Please continue to complete the function. You are not allowed to modify the function "
        "and must provide the completion only. Please return the completed function inside a "
        "code block. Here is the code to complete:\n"
        "```python\n"
        f"{raw_text}\n"
        "```"
    )


def render_prompt_text(tokenizer, user_prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return user_prompt
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def infer_stop_reason(token_ids: list[int], eos_id: int | None, alt_eos_id: int | None, max_new_tokens: int) -> str:
    if eos_id is not None and eos_id in token_ids:
        return "eos"
    if alt_eos_id is not None and alt_eos_id in token_ids:
        return "alt_eos"
    if len(token_ids) >= max_new_tokens:
        return "max_new_tokens"
    return "unknown"


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(args: argparse.Namespace):
    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_hf_model(args: argparse.Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device_map,
        torch_dtype=dtype_from_name(args.dtype),
        attn_implementation=args.attention_impl,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model


def patch_jacobi_method() -> None:
    from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified import (
        jacobi_forward_greedy_multiblock,
    )
    from transformers import Qwen2ForCausalLM

    Qwen2ForCausalLM.jacobi_forward_greedy_multiblock = jacobi_forward_greedy_multiblock
    try:
        from transformers import Qwen3ForCausalLM

        Qwen3ForCausalLM.jacobi_forward_greedy_multiblock = jacobi_forward_greedy_multiblock
    except ImportError:
        pass


def ensure_attention_types(model) -> None:
    layer_types = getattr(model.config, "layer_types", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    for idx, layer in enumerate(layers):
        if hasattr(layer, "attention_type"):
            continue
        if isinstance(layer_types, list) and idx < len(layer_types):
            layer.attention_type = layer_types[idx]
        else:
            layer.attention_type = "full_attention"


def run_hf_example(
    model,
    tokenizer,
    prompt_text: str,
    args: argparse.Namespace,
    task_id: str,
    index: int,
) -> dict[str, Any]:
    device = get_model_device(model)
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
    input_ids = model_inputs["input_ids"]
    prompt_len = int(input_ids.shape[1])

    eos_id = tokenizer.eos_token_id
    alt_eos_id = None if args.alt_eos_id < 0 else args.alt_eos_id

    sync_cuda(args.synchronize_cuda)
    t0 = time.perf_counter()
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=[eos_id, alt_eos_id] if alt_eos_id is not None else eos_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    sync_cuda(args.synchronize_cuda)
    gen_time_sec = time.perf_counter() - t0

    generated_token_ids = output_ids[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    new_tokens = len(generated_token_ids)

    return {
        "backend": "hf",
        "index": index,
        "task_id": task_id,
        "prompt_tokens": prompt_len,
        "new_tokens": new_tokens,
        "gen_time_sec": gen_time_sec,
        "prefill_time_sec": None,
        "decode_time_sec": gen_time_sec,
        "toks_per_sec": (new_tokens / gen_time_sec) if gen_time_sec > 0 else float("nan"),
        "stop_reason": infer_stop_reason(generated_token_ids, eos_id, alt_eos_id, args.max_new_tokens),
        "calls": 1,
        "decode_calls": 1,
        "total_iterations": None,
        "avg_iter_per_decode_call": None,
        "avg_iter_per_token": None,
        "generated_text": generated_text,
    }


def run_jacobi_example(
    model,
    tokenizer,
    prompt_text: str,
    args: argparse.Namespace,
    task_id: str,
    index: int,
) -> dict[str, Any]:
    device = get_model_device(model)
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
    input_ids = model_inputs["input_ids"]
    attention_mask = torch.ones_like(input_ids, device=device)
    prompt_len = int(input_ids.shape[1])

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    alt_eos_id = None if args.alt_eos_id < 0 else args.alt_eos_id

    generated_ids = input_ids.clone()
    past_key_values = None
    prefill_phase = True
    prefill_drafted_n_gram = None
    first_correct_token = None

    calls = 0
    decode_calls = 0
    total_iterations = 0
    total_new_tokens = 0
    prefill_time_sec = 0.0
    decode_time_sec = 0.0
    stop_reason = "unknown"

    while True:
        generated_part = generated_ids[0, prompt_len:].tolist()
        if eos_id is not None and eos_id in generated_part:
            stop_reason = "eos"
            break
        if alt_eos_id is not None and alt_eos_id in generated_part:
            stop_reason = "alt_eos"
            break
        if total_new_tokens >= args.max_new_tokens:
            stop_reason = "max_new_tokens"
            break
        if calls >= args.jacobi_max_calls:
            stop_reason = "max_calls"
            break

        if prefill_phase:
            q_sampled = []
            for _ in range(args.n_token_seq_len):
                q_sample = torch.tensor(
                    [random.choice(generated_ids[0].tolist())],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                q_sampled.append(q_sample)
            prefill_draft_token_ids = torch.cat(q_sampled, dim=1)
            prefill_input_ids = torch.cat((input_ids, prefill_draft_token_ids), dim=-1)

            sync_cuda(args.synchronize_cuda)
            t0 = time.perf_counter()
            past_key_values, first_correct_token, prefill_drafted_n_gram, _ = model.jacobi_forward_greedy_multiblock(
                input_ids=prefill_input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                prefill_phase=True,
                n_token_seq_len=args.n_token_seq_len,
                K=args.jacobi_K,
                r=args.jacobi_r,
                n_gram_pool_size=args.jacobi_n_gram_pool_size,
                lookahead_start_ratio=args.jacobi_lookahead_start_ratio,
                tokenizer=tokenizer,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
            sync_cuda(args.synchronize_cuda)
            prefill_time_sec += time.perf_counter() - t0

            prefill_phase = False
            generated_ids = input_ids
            calls += 1
            continue

        if calls == 1:
            draft_input_ids = prefill_drafted_n_gram
        else:
            q_sampled = []
            for _ in range(max(args.n_token_seq_len - 1, 1)):
                q_sample = torch.tensor(
                    [random.choice(generated_ids[0].tolist())],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                q_sampled.append(q_sample)
            q_sampled = torch.cat(q_sampled, dim=1)
            draft_input_ids = torch.cat((first_correct_token.view(1, -1), q_sampled), dim=-1)

        sync_cuda(args.synchronize_cuda)
        t0 = time.perf_counter()
        past_key_values, first_correct_token, accepted_n_gram, itr_count = model.jacobi_forward_greedy_multiblock(
            input_ids=draft_input_ids,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            prefill_phase=False,
            n_token_seq_len=args.n_token_seq_len,
            K=args.jacobi_K,
            r=args.jacobi_r,
            n_gram_pool_size=args.jacobi_n_gram_pool_size,
            lookahead_start_ratio=args.jacobi_lookahead_start_ratio,
            tokenizer=tokenizer,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        sync_cuda(args.synchronize_cuda)
        decode_time_sec += time.perf_counter() - t0

        calls += 1
        decode_calls += 1
        total_iterations += int(itr_count)

        if accepted_n_gram is None or accepted_n_gram.numel() == 0:
            continue

        generated_ids = torch.cat((generated_ids, accepted_n_gram), dim=-1)
        total_new_tokens = int(generated_ids.shape[1] - prompt_len)

    generated_token_ids = generated_ids[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    gen_time_sec = prefill_time_sec + decode_time_sec

    return {
        "backend": "jacobi",
        "index": index,
        "task_id": task_id,
        "prompt_tokens": prompt_len,
        "new_tokens": len(generated_token_ids),
        "gen_time_sec": gen_time_sec,
        "prefill_time_sec": prefill_time_sec,
        "decode_time_sec": decode_time_sec,
        "toks_per_sec": (len(generated_token_ids) / gen_time_sec) if gen_time_sec > 0 else float("nan"),
        "stop_reason": stop_reason,
        "calls": calls,
        "decode_calls": decode_calls,
        "total_iterations": total_iterations,
        "avg_iter_per_decode_call": (total_iterations / decode_calls) if decode_calls > 0 else float("nan"),
        "avg_iter_per_token": (total_iterations / len(generated_token_ids)) if generated_token_ids else float("nan"),
        "generated_text": generated_text,
    }


def build_vllm_llm(args: argparse.Namespace):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requested but `vllm` is not installed in the active environment."
        ) from exc

    llm_kwargs = {
        "model": args.model_name,
        "tokenizer": args.tokenizer_name or args.model_name,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    return LLM(**llm_kwargs)


def build_nano_vllm_llm(args: argparse.Namespace, *, jacobi_enabled: bool):
    from inference_engine import LLM

    llm_kwargs = {
        "tokenizer_path": args.tokenizer_name or args.model_name,
        "enforce_eager": args.nano_vllm_enforce_eager,
        "tensor_parallel_size": args.tensor_parallel_size,
        "jacobi_enabled": jacobi_enabled,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    return LLM(args.model_name, **llm_kwargs)


def reset_nano_vllm_jacobi_stats(llm) -> None:
    jacobi_decoder = getattr(llm.model_runner, "jacobi_decoder", None)
    if jacobi_decoder is not None:
        jacobi_decoder.stats = {
            "num_chunk_calls": 0,
            "num_jacobi_iterations": 0,
            "tokens_accepted": 0,
            "tokens_per_call": [],
            "tokens_per_iteration": [],
            "iterations_per_call": [],
        }


def run_nano_vllm_example(
    llm,
    tokenizer,
    prompt_text: str,
    args: argparse.Namespace,
    task_id: str,
    index: int,
    *,
    decode_strategy: str,
) -> dict[str, Any]:
    from inference_engine import SamplingParams

    prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_token_ids)

    if decode_strategy == "jacobi":
        reset_nano_vllm_jacobi_stats(llm)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        decode_strategy=decode_strategy,
        jacobi_block_len=args.n_token_seq_len,
        jacobi_max_iterations=args.jacobi_max_calls,
        jacobi_max_blocks=args.jacobi_K,
        jacobi_spawn_ratio=args.jacobi_r,
        jacobi_lookahead_start_ratio=args.jacobi_lookahead_start_ratio,
        jacobi_n_gram_pool_size=args.jacobi_n_gram_pool_size,
    )

    t0 = time.perf_counter()
    outputs = llm.generate([prompt_text], sampling_params, use_tqdm=False)
    gen_time_sec = time.perf_counter() - t0

    output = outputs[0]
    generated_text = output["text"]
    generated_token_ids = list(output["token_ids"])
    new_tokens = len(generated_token_ids)

    jacobi_stats = None
    total_iterations = None
    calls = 1
    decode_calls = 1
    avg_iter_per_decode_call = None
    avg_iter_per_token = None

    if decode_strategy == "jacobi":
        jacobi_decoder = getattr(llm.model_runner, "jacobi_decoder", None)
        jacobi_stats = jacobi_decoder.stats.copy() if jacobi_decoder is not None else {}
        calls = int(jacobi_stats.get("num_chunk_calls", 0) or 0)
        decode_calls = calls
        total_iterations = int(jacobi_stats.get("num_jacobi_iterations", 0) or 0)
        avg_iter_per_decode_call = (
            total_iterations / decode_calls if decode_calls > 0 else float("nan")
        )
        avg_iter_per_token = total_iterations / new_tokens if new_tokens > 0 else float("nan")

    eos_id = tokenizer.eos_token_id
    alt_eos_id = None if args.alt_eos_id < 0 else args.alt_eos_id

    return {
        "backend": "nano_vllm_jacobi" if decode_strategy == "jacobi" else "nano_vllm_ar",
        "index": index,
        "task_id": task_id,
        "prompt_tokens": prompt_len,
        "new_tokens": new_tokens,
        "gen_time_sec": gen_time_sec,
        "prefill_time_sec": None,
        "decode_time_sec": gen_time_sec,
        "toks_per_sec": (new_tokens / gen_time_sec) if gen_time_sec > 0 else float("nan"),
        "stop_reason": infer_stop_reason(generated_token_ids, eos_id, alt_eos_id, args.max_new_tokens),
        "calls": calls,
        "decode_calls": decode_calls,
        "total_iterations": total_iterations,
        "avg_iter_per_decode_call": avg_iter_per_decode_call,
        "avg_iter_per_token": avg_iter_per_token,
        "generated_text": generated_text,
    }


def run_nano_vllm_ar_example(llm, tokenizer, prompt_text: str, args: argparse.Namespace, task_id: str, index: int):
    return run_nano_vllm_example(
        llm, tokenizer, prompt_text, args, task_id, index, decode_strategy="autoregressive"
    )


def run_nano_vllm_jacobi_example(
    llm, tokenizer, prompt_text: str, args: argparse.Namespace, task_id: str, index: int
):
    return run_nano_vllm_example(
        llm, tokenizer, prompt_text, args, task_id, index, decode_strategy="jacobi"
    )


def run_vllm_example(
    llm,
    tokenizer,
    prompt_text: str,
    args: argparse.Namespace,
    task_id: str,
    index: int,
) -> dict[str, Any]:
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM backend requested but `vllm` is not installed.") from exc

    tokenized = tokenizer(prompt_text, add_special_tokens=False, return_tensors=None)
    prompt_token_ids = tokenized["input_ids"]
    prompt_len = len(prompt_token_ids)
    eos_id = tokenizer.eos_token_id
    alt_eos_id = None if args.alt_eos_id < 0 else args.alt_eos_id

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    sync_cuda(args.synchronize_cuda)
    t0 = time.perf_counter()
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": prompt_token_ids, "prompt": prompt_text}],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    sync_cuda(args.synchronize_cuda)
    gen_time_sec = time.perf_counter() - t0

    request_output = outputs[0]
    completion = request_output.outputs[0]
    generated_token_ids = list(getattr(completion, "token_ids", []) or [])
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    stop_reason = getattr(completion, "finish_reason", None) or infer_stop_reason(
        generated_token_ids, eos_id, alt_eos_id, args.max_new_tokens
    )
    new_tokens = len(generated_token_ids)

    return {
        "backend": "vllm",
        "index": index,
        "task_id": task_id,
        "prompt_tokens": prompt_len,
        "new_tokens": new_tokens,
        "gen_time_sec": gen_time_sec,
        "prefill_time_sec": None,
        "decode_time_sec": gen_time_sec,
        "toks_per_sec": (new_tokens / gen_time_sec) if gen_time_sec > 0 else float("nan"),
        "stop_reason": stop_reason,
        "calls": 1,
        "decode_calls": 1,
        "total_iterations": None,
        "avg_iter_per_decode_call": None,
        "avg_iter_per_token": None,
        "generated_text": generated_text,
    }


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    df_eos = df[df["stop_reason"].isin(["eos", "alt_eos"])].copy()
    target = df_eos if not df_eos.empty else df
    summary = {
        "num_rows": int(len(df)),
        "num_eos_rows": int(len(df_eos)),
        "sum_new_tokens": float(pd.to_numeric(df["new_tokens"], errors="coerce").fillna(0).sum()),
        "sum_gen_time_sec": float(pd.to_numeric(df["gen_time_sec"], errors="coerce").fillna(0).sum()),
        "avg_prompt_tokens": float(pd.to_numeric(target["prompt_tokens"], errors="coerce").mean()),
        "avg_new_tokens": float(pd.to_numeric(target["new_tokens"], errors="coerce").mean()),
        "avg_gen_time_sec": float(pd.to_numeric(target["gen_time_sec"], errors="coerce").mean()),
        "avg_toks_per_sec": float(pd.to_numeric(target["toks_per_sec"], errors="coerce").mean()),
        "p50_toks_per_sec": float(pd.to_numeric(target["toks_per_sec"], errors="coerce").median()),
        "overall_toks_per_sec": float(
            pd.to_numeric(df["new_tokens"], errors="coerce").fillna(0).sum()
            / max(pd.to_numeric(df["gen_time_sec"], errors="coerce").fillna(0).sum(), 1e-12)
        ),
        "stop_reasons": df["stop_reason"].value_counts(dropna=False).to_dict(),
    }
    if "calls" in df.columns:
        summary["avg_calls"] = float(pd.to_numeric(target["calls"], errors="coerce").mean())
    if "total_iterations" in df.columns:
        valid = pd.to_numeric(target["total_iterations"], errors="coerce")
        if not valid.isna().all():
            summary["avg_total_iterations"] = float(valid.mean())
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Offline Benchmark Summary ===")
    print(f"Measured rows: {summary['num_rows']}  EOS rows: {summary['num_eos_rows']}")
    print(f"Avg prompt tokens: {summary['avg_prompt_tokens']:.2f}")
    print(f"Avg new tokens: {summary['avg_new_tokens']:.2f}")
    print(f"Avg gen time: {summary['avg_gen_time_sec']:.4f}s")
    print(f"Avg toks/sec: {summary['avg_toks_per_sec']:.2f}")
    print(f"P50 toks/sec: {summary['p50_toks_per_sec']:.2f}")
    print(f"Overall toks/sec: {summary['overall_toks_per_sec']:.2f}")
    if "avg_calls" in summary:
        print(f"Avg calls: {summary['avg_calls']:.2f}")
    if "avg_total_iterations" in summary:
        print(f"Avg total iterations: {summary['avg_total_iterations']:.2f}")
    print("Stop reasons:")
    for key, value in summary["stop_reasons"].items():
        print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    set_seed(args.seed)

    tokenizer = load_tokenizer(args)
    records = load_records(args.dataset, limit=0)
    warmup_count = min(args.warmup, len(records))
    measured_records = records[warmup_count:]
    if args.limit > 0:
        measured_records = measured_records[: args.limit]

    print(f"Loaded {len(records)} records from {args.dataset}")
    print(f"Warmup rows: {warmup_count}")
    print(f"Measured rows: {len(measured_records)}")

    backend_runner = None
    backend_obj = None
    if args.backend == "hf":
        backend_obj = load_hf_model(args)
        backend_runner = run_hf_example
    elif args.backend == "jacobi":
        patch_jacobi_method()
        backend_obj = load_hf_model(args)
        ensure_attention_types(backend_obj)
        backend_runner = run_jacobi_example
    elif args.backend == "vllm":
        backend_obj = build_vllm_llm(args)
        backend_runner = run_vllm_example
    elif args.backend == "nano_vllm_ar":
        backend_obj = build_nano_vllm_llm(args, jacobi_enabled=False)
        backend_runner = run_nano_vllm_ar_example
    else:
        backend_obj = build_nano_vllm_llm(args, jacobi_enabled=True)
        backend_runner = run_nano_vllm_jacobi_example

    warmup_records_list = records[:warmup_count]
    if warmup_records_list:
        print("Running warmup...")
        for idx, row in enumerate(tqdm(warmup_records_list, total=len(warmup_records_list), desc="warmup")):
            raw_prompt = row[args.text_key]
            task_id = row.get(args.task_id_key, f"warmup_{idx}")
            user_prompt = build_prompt(raw_prompt, args.prompt_format)
            prompt_text = render_prompt_text(tokenizer, user_prompt, args.use_chat_template)
            backend_runner(backend_obj, tokenizer, prompt_text, args, task_id, idx)
        set_seed(args.seed)

    rows: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    for idx, row in enumerate(tqdm(measured_records, total=len(measured_records), desc=args.backend)):
        raw_prompt = row[args.text_key]
        task_id = row.get(args.task_id_key, f"idx_{idx}")
        user_prompt = build_prompt(raw_prompt, args.prompt_format)
        prompt_text = render_prompt_text(tokenizer, user_prompt, args.use_chat_template)

        result = backend_runner(backend_obj, tokenizer, prompt_text, args, task_id, idx)
        rows.append({k: v for k, v in result.items() if k != "generated_text"})
        if args.output_generations_jsonl:
            generations.append(
                {
                    "index": idx,
                    "task_id": task_id,
                    "backend": args.backend,
                    "prompt": raw_prompt,
                    "generated_text": result["generated_text"],
                }
            )

        if args.print_every > 0 and ((idx + 1) % args.print_every == 0 or (idx + 1) == len(measured_records)):
            print(
                f"[{idx + 1}/{len(measured_records)}] task_id={task_id} "
                f"new_tokens={result['new_tokens']} gen_time={result['gen_time_sec']:.4f}s "
                f"toks/sec={result['toks_per_sec']:.2f} stop={result['stop_reason']}"
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved benchmark CSV to {output_csv}")

    if args.output_generations_jsonl:
        output_jsonl = Path(args.output_generations_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as fh:
            for row in generations:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved generations JSONL to {output_jsonl}")

    summary = summarize(df)
    print_summary(summary)

    summary_path = output_csv.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"Saved summary JSON to {summary_path}")

    del backend_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
