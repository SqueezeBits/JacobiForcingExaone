#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate K-EXAONE training trajectories with vLLM.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input-file", required=True, help="Bucket JSON or JSONL file containing prompts")
    parser.add_argument("--output-file", required=True, help="Output trajectory JSON")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--refinement-steps", type=int, default=8)
    parser.add_argument("--system-prompt", default="You are a helpful coding assistant.")
    return parser.parse_args()


def load_prompts(path: str) -> list[str]:
    p = Path(path)
    if p.suffix == ".jsonl":
        prompts = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if isinstance(obj, str):
                    prompts.append(obj)
                else:
                    prompts.append(obj.get("input") or obj.get("prompt") or obj.get("text") or "")
        return prompts
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    prompts = []
    for obj in data:
        if isinstance(obj, str):
            prompts.append(obj)
        else:
            prompts.append(obj.get("input") or obj.get("prompt") or obj.get("text") or "")
    return prompts


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_prompt(tokenizer, user_text: str, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def synthesize_block_trajectory(
    teacher_block_ids: list[int],
    prefix_ids: list[int],
    block_size: int,
    refinement_steps: int,
    pad_id: int,
) -> list[list[int]]:
    if len(teacher_block_ids) < block_size:
        teacher_block_ids = teacher_block_ids + [pad_id] * (block_size - len(teacher_block_ids))
    else:
        teacher_block_ids = teacher_block_ids[:block_size]

    source = prefix_ids if prefix_ids else [pad_id]
    states: list[list[int]] = []
    initial = source[-block_size:]
    if len(initial) < block_size:
        initial = initial + [pad_id] * (block_size - len(initial))
    states.append(list(initial[:block_size]))

    reveal_schedule = sorted(
        {
            max(1, min(block_size, int(round(block_size * (i / max(1, refinement_steps - 1))))))
            for i in range(1, refinement_steps)
        }
    )
    for reveal in reveal_schedule:
        state = teacher_block_ids[:reveal] + [pad_id] * (block_size - reveal)
        states.append(state)
    if states[-1] != teacher_block_ids:
        states.append(list(teacher_block_ids))
    return states


def main() -> int:
    args = parse_args()
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "vLLM is required for this script. Install it via pixi or pip before running."
        ) from exc

    prompts = load_prompts(args.input_file)
    if args.max_samples > 0:
        prompts = prompts[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)

    rendered_prompts = [build_prompt(tokenizer, prompt, args.system_prompt) for prompt in prompts]
    teacher_outputs = llm.generate(rendered_prompts, sampling_params)

    records = []
    for idx, (rendered, teacher) in enumerate(zip(rendered_prompts, teacher_outputs)):
        prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        teacher_text = teacher.outputs[0].text if getattr(teacher, "outputs", None) else str(teacher)
        teacher_completion_ids = tokenizer(teacher_text, add_special_tokens=False)["input_ids"]
        teacher_output_ids = prompt_ids + teacher_completion_ids

        for block_idx in range(0, len(teacher_completion_ids), args.block_size):
            block_tokens = teacher_completion_ids[block_idx : block_idx + args.block_size]
            prefix_ids = prompt_ids + teacher_completion_ids[:block_idx]
            trajectory_states = synthesize_block_trajectory(
                block_tokens,
                prefix_ids=prefix_ids,
                block_size=args.block_size,
                refinement_steps=args.refinement_steps,
                pad_id=pad_id,
            )
            records.append(
                {
                    "diffusion_itr_id": f"itr_{block_idx // args.block_size}",
                    "data_id": f"data_{idx}",
                    "prompt_ids": [prefix_ids],
                    "answer_trajectory_ids": trajectory_states,
                    "teacher_output_ids": teacher_output_ids,
                }
            )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(json.dumps({"output_file": str(output_path), "num_records": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
