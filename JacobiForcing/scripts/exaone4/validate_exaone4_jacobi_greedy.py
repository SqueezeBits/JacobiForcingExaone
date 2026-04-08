#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.exaone4.modeling_exaone4 import Exaone4ForCausalLM


ROOT = "/workspace/exaone_workspace/JacobiForcing"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from generate_trajectory.generation.exaone4_modeling_jacobi_forcing_greedy import (  # noqa: E402
    get_jacobi_forward_trajectory_greedy,
)


DEFAULT_PROMPT = (
    "Write a short Python function named fib(n) that returns the nth Fibonacci number. "
    "Return code only."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EXAONE4 Jacobi greedy block against AR greedy.")
    parser.add_argument("--model-id", default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attn-implementation", default="flex_attention")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    Exaone4ForCausalLM.get_jacobi_forward_trajectory_greedy = get_jacobi_forward_trajectory_greedy

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": args.prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(rendered, return_tensors="pt").to(device)

    prompt_input_ids = model_inputs["input_ids"]
    prompt_len = prompt_input_ids.shape[1]

    with torch.inference_mode():
        ar_output = model.generate(
            **model_inputs,
            max_new_tokens=args.block_size,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    ar_block = ar_output[:, prompt_len : prompt_len + args.block_size]

    past_key_values, first_correct_token = model.get_jacobi_forward_trajectory_greedy(
        input_ids=prompt_input_ids,
        attention_mask=model_inputs["attention_mask"],
        past_key_values=None,
        use_cache=True,
        prefill_phase=True,
        n_token_seq_len=args.block_size,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
    )

    candidate_pool = prompt_input_ids[0].tolist()
    sampled_tail = random.choices(candidate_pool, k=max(args.block_size - 1, 0))
    sampled_tail = torch.tensor(sampled_tail, dtype=torch.long, device=device).unsqueeze(0)
    draft_input_ids = torch.cat((first_correct_token.view(1, -1), sampled_tail), dim=-1)

    past_key_values, next_token, answer_trajectory_ids = model.get_jacobi_forward_trajectory_greedy(
        input_ids=draft_input_ids,
        attention_mask=None,
        past_key_values=past_key_values,
        use_cache=True,
        prefill_phase=False,
        n_token_seq_len=args.block_size,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
    )

    final_block = answer_trajectory_ids[-1]
    ar_cmp = ar_block[:, : final_block.shape[1]]
    exact_match = bool(torch.equal(final_block, ar_cmp))

    print(json.dumps(
        {
            "block_size": args.block_size,
            "num_trajectory_steps": len(answer_trajectory_ids),
            "first_correct_token": int(first_correct_token.item()),
            "final_block_len": int(final_block.shape[1]),
            "ar_block_len": int(ar_block.shape[1]),
            "exact_match_with_ar_prefix": exact_match,
            "final_block_text": tokenizer.decode(final_block[0], skip_special_tokens=False),
            "ar_block_text": tokenizer.decode(ar_cmp[0], skip_special_tokens=False),
            "trajectory_texts": [
                tokenizer.decode(step[0], skip_special_tokens=False) for step in answer_trajectory_ids
            ],
        },
        indent=2,
        ensure_ascii=False,
    ))

    return 0 if exact_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
