#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

path_root = Path(__file__).resolve().parents[1]
sys.path.append(str(path_root))

from qwen2_modeling_jacobi_forcing_greedy import (  # noqa: E402
    get_jacobi_forward_trajectory_greedy as qwen_get_jacobi_forward_trajectory_greedy,
)
from solar_modeling_jacobi_forcing_greedy import (  # noqa: E402
    get_jacobi_forward_trajectory_greedy as solar_get_jacobi_forward_trajectory_greedy,
)

# Suppress a noisy third-party deprecation warning emitted by nvidia_cutlass_dsl during model load.
warnings.filterwarnings(
    "ignore",
    message=r"Use explicit `struct\.scalar\.ptr` for pointer instead\.",
    category=DeprecationWarning,
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        return rank, world_size, torch.device(f"cuda:{local_rank}")

    if torch.cuda.is_available():
        return 0, 1, torch.device("cuda:0")
    return 0, 1, torch.device("cpu")


def cleanup_distributed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def load_records(filename: str, start: int = 0, end: int | None = None) -> List[Dict[str, Any]]:
    with open(filename, "r", encoding="utf-8") as fin:
        payload = json.load(fin)
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level JSON list in {filename}")

    end = len(payload) if end is None else min(end, len(payload))
    selected = payload[start:end]
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected, start=start):
        if isinstance(item, str):
            normalized.append({"data_id": f"data_{idx}", "prompt": item})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Record {idx} in {filename} is not a dict")
        normalized.append(item)
    return normalized


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


def infer_model_family(model) -> str:
    model_type = str(getattr(model.config, "model_type", "")).lower()
    if "qwen" in model_type:
        return "qwen"
    return "solar"


def attach_jacobi_forward(model) -> str:
    ensure_attention_types(model)
    model_family = infer_model_family(model)
    if model_family == "qwen":
        patch_fn = qwen_get_jacobi_forward_trajectory_greedy
    else:
        patch_fn = solar_get_jacobi_forward_trajectory_greedy
    setattr(type(model), "get_jacobi_forward_trajectory_greedy", patch_fn)
    return model_family


def build_generation_prompt(tokenizer, prompt: str, chat_template_mode: str) -> str:
    if chat_template_mode == "solar":
        messages = [{"role": "user", "content": prompt}]
    elif chat_template_mode == "qwen":
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        raise ValueError(f"Unsupported --chat_template_mode: {chat_template_mode}")

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def append_skip(skip_records: List[Dict[str, Any]], data_id: str, reason: str, **extra: Any) -> None:
    record = {"data_id": data_id, "reason": reason}
    record.update(extra)
    skip_records.append(record)


def sample_draft_tokens(generated_ids: torch.Tensor, n_token_seq_len: int, rng: random.Random) -> torch.Tensor:
    population = generated_ids[0].tolist()
    sampled = rng.choices(population, k=n_token_seq_len - 1)
    return torch.tensor(sampled, dtype=torch.long, device=generated_ids.device).unsqueeze(0)


def process_record(
    record: Dict[str, Any],
    record_idx: int,
    model,
    tokenizer,
    device: torch.device,
    n_token_seq_len: int,
    max_new_seq_len: int,
    chat_template_mode: str,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # One record corresponds to one prompt. We keep extending `generated_ids` block by block
    # and store every Jacobi refinement trajectory as a separate training row.
    data_id = str(record.get("data_id", f"data_{record_idx}"))
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return [], [{"data_id": data_id, "reason": "missing_prompt"}]

    skip_records: List[Dict[str, Any]] = []
    prompt_text = build_generation_prompt(tokenizer, prompt, chat_template_mode)
    model_inputs = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True).to(device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    generated_ids = input_ids.clone()

    if generated_ids.ndim != 2 or generated_ids.shape[0] != 1:
        append_skip(skip_records, data_id, "invalid_prompt_shape", prompt_shape=list(generated_ids.shape))
        return [], skip_records

    iterations = 0
    prefill_phase = True
    past_key_values = None
    first_correct_token = None
    per_iteration_records: List[Dict[str, Any]] = []
    rng = random.Random(seed + record_idx)

    while True:
        # Stop once the answer has emitted EOS or we hit the configured maximum generated length.
        generated_part = generated_ids[:, input_ids.size(1) :]
        if tokenizer.eos_token_id is not None and (generated_part == tokenizer.eos_token_id).any():
            break
        if iterations * n_token_seq_len >= max_new_seq_len:
            break

        if prefill_phase:
            past_key_values, first_correct_token = model.get_jacobi_forward_trajectory_greedy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                prefill_phase=True,
                n_token_seq_len=n_token_seq_len,
                tokenizer=tokenizer,
                eos_token_id=tokenizer.eos_token_id,
            )
            prefill_phase = False
            continue

        if first_correct_token is None:
            append_skip(skip_records, data_id, "missing_first_correct_token", diffusion_itr_id=f"itr_{iterations}")
            break

        # Build the next Jacobi draft block from the fixed first token plus random tokens sampled
        # from the already-generated prefix. The model-specific Jacobi forward will then refine it.
        prompt_ids_for_record = generated_ids.clone()
        q_sampled = sample_draft_tokens(generated_ids, n_token_seq_len, rng)
        jacobi_input_ids = torch.cat((first_correct_token.view(1, -1), q_sampled), dim=-1)
        past_key_values, first_correct_token, answer_trajectory_ids = model.get_jacobi_forward_trajectory_greedy(
            input_ids=jacobi_input_ids,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            prefill_phase=False,
            n_token_seq_len=n_token_seq_len,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
        )

        if not answer_trajectory_ids:
            append_skip(skip_records, data_id, "empty_answer_trajectory", diffusion_itr_id=f"itr_{iterations}")
            break

        final_block = answer_trajectory_ids[-1]
        if final_block.ndim != 2 or final_block.shape[0] != 1:
            append_skip(
                skip_records,
                data_id,
                "invalid_answer_trajectory_shape",
                diffusion_itr_id=f"itr_{iterations}",
                answer_shape=list(final_block.shape),
            )
            break

        final_block_len = int(final_block.shape[-1])
        if final_block_len != n_token_seq_len:
            append_skip(
                skip_records,
                data_id,
                "short_final_block",
                diffusion_itr_id=f"itr_{iterations}",
                final_block_len=final_block_len,
                expected_block_len=n_token_seq_len,
            )
            break

        generated_ids = torch.cat((generated_ids, final_block), dim=-1)
        # Each diffusion iteration becomes one row in the trajectory dataset.
        per_iteration_records.append(
            {
                "diffusion_itr_id": f"itr_{iterations}",
                "data_id": data_id,
                "prompt_ids": prompt_ids_for_record.cpu().tolist(),
                "answer_trajectory_ids": [step[0].cpu().tolist() for step in answer_trajectory_ids],
            }
        )
        iterations += 1

    if not per_iteration_records:
        if not skip_records:
            append_skip(skip_records, data_id, "no_valid_iterations")
        return [], skip_records

    teacher_output_ids = generated_ids[0].cpu().tolist()
    for iteration_record in per_iteration_records:
        iteration_record["teacher_output_ids"] = teacher_output_ids

    return per_iteration_records, skip_records


def main(args):
    set_random_seed(args.seed)
    rank, world_size, device = setup_distributed()

    try:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", #"flash_attention_4",
            trust_remote_code=True,
        )
        if world_size > 1:
            model_kwargs["tp_plan"] = "auto"
        else:
            model_kwargs["device_map"] = "cuda" if device.type == "cuda" else None
        if rank == 0:
            print(
                f"Loading model={args.model} tokenizer={args.tokenizer_path} "
                f"n_token_seq_len={args.n_token_seq_len} max_new_seq_len={args.max_new_seq_len}"
            )
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_family = attach_jacobi_forward(model)
        if rank == 0:
            print(f"Attached Jacobi forward for model_family={model_family}")
        records = load_records(
            args.filename,
            start=int(args.data_bos_id),
            end=None if int(args.data_eos_id) < 0 else int(args.data_eos_id),
        )

        generated_records: List[Dict[str, Any]] = []
        skip_records: List[Dict[str, Any]] = []
        for record_idx, record in enumerate(
            tqdm(records, desc="Generating trajectories", total=len(records), disable=rank != 0)
        ):
            sample_records, sample_skips = process_record(
                record=record,
                record_idx=record_idx + int(args.data_bos_id),
                model=model,
                tokenizer=tokenizer,
                device=device,
                n_token_seq_len=args.n_token_seq_len,
                max_new_seq_len=args.max_new_seq_len,
                chat_template_mode=args.chat_template_mode,
                seed=args.seed,
            )
            if rank == 0:
                generated_records.extend(sample_records)
                skip_records.extend(sample_skips)

        if world_size > 1:
            dist.barrier()

        if rank == 0:
            os.makedirs(args.save_path, exist_ok=True)
            stem = Path(args.filename).stem
            range_suffix = f"{int(args.data_bos_id)}_{int(args.data_eos_id)}"
            output_file = os.path.join(
                args.save_path,
                f"{stem}_{args.chat_template_mode}_{model_family}_greedy_jacobi_len{args.n_token_seq_len}_{range_suffix}.json",
            )
            skip_file = os.path.join(
                args.save_path,
                f"{stem}_{args.chat_template_mode}_{model_family}_greedy_jacobi_len{args.n_token_seq_len}_{range_suffix}_skipped.jsonl",
            )

            with open(output_file, "w", encoding="utf-8") as fout:
                json.dump(generated_records, fout, ensure_ascii=False)
            with open(skip_file, "w", encoding="utf-8") as fout:
                for skip_record in skip_records:
                    fout.write(json.dumps(skip_record, ensure_ascii=False) + "\n")

            print(f"Wrote {len(generated_records)} trajectory records to {output_file}")
            print(f"Wrote {len(skip_records)} skip records to {skip_file}")
    finally:
        cleanup_distributed(world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_token_seq_len", type=int, default=64)
    parser.add_argument("--max_new_seq_len", type=int, default=16384)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--data_start_id", default=0)
    parser.add_argument("--data_bos_id", default=0)
    parser.add_argument("--data_eos_id", default=-1)
    parser.add_argument("--use_labels", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat_template_mode", default="solar", choices=["solar", "qwen"])
    args = parser.parse_args()

    main(args)
