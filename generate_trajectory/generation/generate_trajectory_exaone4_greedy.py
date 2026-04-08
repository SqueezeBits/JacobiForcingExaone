#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.exaone4.modeling_exaone4 import Exaone4ForCausalLM

import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from exaone4_modeling_jacobi_forcing_greedy import get_jacobi_forward_trajectory_greedy


Exaone4ForCausalLM.get_jacobi_forward_trajectory_greedy = get_jacobi_forward_trajectory_greedy


DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."


def load_prompt_list(filename: str, start: int = 0, end: int | None = None) -> list[str]:
    path = Path(filename)
    if path.suffix == ".jsonl":
        items: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if isinstance(obj, str):
                    items.append(obj)
                elif isinstance(obj, dict):
                    items.append(obj.get("input") or obj.get("prompt") or obj.get("text") or "")
                else:
                    raise ValueError(f"Unsupported JSONL row type: {type(obj)}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {filename}")
        items = []
        for obj in data:
            if isinstance(obj, str):
                items.append(obj)
            elif isinstance(obj, dict):
                items.append(obj.get("input") or obj.get("prompt") or obj.get("text") or "")
            else:
                raise ValueError(f"Unsupported JSON item type: {type(obj)}")

    end = len(items) if end is None else min(end, len(items))
    return items[start:end]


def build_prompt(tokenizer, user_text: str, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def decode_preview(tokenizer, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def pad_state_to_block_len(state: list[int], block_len: int, pad_id: int) -> list[int]:
    if len(state) >= block_len:
        return list(state[:block_len])
    return list(state) + [pad_id] * (block_len - len(state))


def pad_sequences_right(seqs: list[list[int]], pad_id: int, device: torch.device):
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    input_ids = torch.full((len(seqs), max_len), fill_value=pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        seq_t = torch.tensor(seq, dtype=torch.long, device=device)
        input_ids[i, : len(seq)] = seq_t
        attention_mask[i, : len(seq)] = 1
    return input_ids, attention_mask, lengths


@torch.inference_mode()
def prefill_first_correct_tokens_batch(
    model,
    prompt_token_lists: list[list[int]],
    pad_id: int,
) -> list[int]:
    device = model.device
    input_ids, attention_mask, lengths = pad_sequences_right(prompt_token_lists, pad_id, device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits.float()
    first_tokens: list[int] = []
    for i, prompt_len in enumerate(lengths):
        token_id = int(torch.argmax(logits[i, prompt_len - 1, :]).item())
        first_tokens.append(token_id)
    return first_tokens


@torch.inference_mode()
def generate_one_block_trajectory_batch(
    model,
    prefixes: list[list[int]],
    seed_tokens: list[int],
    block_len: int,
    pad_id: int,
    eos_id: int | None,
) -> list[dict]:
    """
    Batched, stateless Jacobi trajectory generation for one block.

    This intentionally rebuilds the full sequence each Jacobi iteration:
      full_input = committed_prefix + current_draft_state

    It is less cache-efficient than the single-sample cache path, but it is
    much easier to batch and tends to improve GPU utilization substantially.
    """
    B = len(prefixes)
    device = model.device

    accepted: list[list[int]] = [[] for _ in range(B)]
    drafts: list[list[int]] = []
    trajectories: list[list[list[int]]] = [[] for _ in range(B)]
    stop_hits = [False] * B
    finished = [False] * B

    for i in range(B):
        tail = random.choices(prefixes[i], k=max(block_len - 1, 0))
        draft = [int(seed_tokens[i])] + [int(t) for t in tail]
        drafts.append(draft)
        trajectories[i].append(pad_state_to_block_len(draft, block_len, pad_id))

    while not all(finished):
        active_indices = [i for i, done in enumerate(finished) if not done]
        full_inputs = [prefixes[i] + drafts[i] for i in active_indices]
        input_ids, attention_mask, _ = pad_sequences_right(full_inputs, pad_id, device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        batch_logits = outputs.logits.float()

        for local_idx, sample_idx in enumerate(active_indices):
            prefix_len = len(prefixes[sample_idx])
            draft = drafts[sample_idx]
            L = len(draft)
            local_logits = batch_logits[local_idx, prefix_len : prefix_len + L, :]

            if L == 1:
                num_accepted_raw = 1
            else:
                greedy_tokens = torch.argmax(local_logits[:-1, :], dim=-1).tolist()
                mismatch = [int(draft[pos + 1]) != int(greedy_tokens[pos]) for pos in range(L - 1)]
                num_matches = 0
                for mis in mismatch:
                    if mis:
                        break
                    num_matches += 1
                num_accepted_raw = num_matches + 1

            num_accepted = num_accepted_raw
            draft_prefix = draft[:num_accepted_raw]

            if eos_id is not None:
                for pos, tok in enumerate(draft_prefix):
                    if int(tok) == int(eos_id):
                        num_accepted = pos + 1
                        stop_hits[sample_idx] = True
                        break

            if num_accepted > 0:
                accepted[sample_idx].extend(int(t) for t in draft[:num_accepted])

            if len(accepted[sample_idx]) >= block_len:
                accepted[sample_idx] = accepted[sample_idx][:block_len]
                trajectories[sample_idx].append(
                    pad_state_to_block_len(accepted[sample_idx], block_len, pad_id)
                )
                finished[sample_idx] = True
                continue

            if stop_hits[sample_idx]:
                trajectories[sample_idx].append(
                    pad_state_to_block_len(accepted[sample_idx], block_len, pad_id)
                )
                finished[sample_idx] = True
                continue

            has_rejected = num_accepted_raw < L
            if has_rejected:
                next_token = int(torch.argmax(local_logits[num_accepted_raw - 1, :]).item())
                rebuilt = [next_token]

                if num_accepted_raw < L - 1:
                    greedy_tail = torch.argmax(local_logits[num_accepted_raw : L - 1, :], dim=-1).tolist()
                    rebuilt.extend(int(t) for t in greedy_tail)

                drafts[sample_idx] = rebuilt
                visible_state = accepted[sample_idx] + rebuilt
                trajectories[sample_idx].append(
                    pad_state_to_block_len(visible_state, block_len, pad_id)
                )
                continue

            next_token = int(torch.argmax(local_logits[L - 1, :]).item())
            accepted[sample_idx].append(next_token)
            trajectories[sample_idx].append(
                pad_state_to_block_len(accepted[sample_idx], block_len, pad_id)
            )

            if (eos_id is not None and next_token == int(eos_id)) or len(accepted[sample_idx]) >= block_len:
                if eos_id is not None and next_token == int(eos_id):
                    stop_hits[sample_idx] = True
                accepted[sample_idx] = accepted[sample_idx][:block_len]
                finished[sample_idx] = True
            else:
                drafts[sample_idx] = [next_token]

    results: list[dict] = []
    for i in range(B):
        committed_block = accepted[i][:block_len]
        if committed_block:
            next_seed = int(committed_block[-1])
        else:
            next_seed = int(seed_tokens[i])
        results.append(
            {
                "trajectory_states": trajectories[i],
                "committed_block": committed_block,
                "next_seed": next_seed,
                "stop_hit": bool(stop_hits[i]),
            }
        )
    return results


def main(
    filename: str,
    model,
    tokenizer,
    n_token_seq_len: int,
    max_new_seq_len: int,
    data_bos_id: int,
    data_eos_id: int,
    save_path: str,
    system_prompt: str,
    batch_size: int,
):
    m = re.search(r"bucket_(\d+)", filename)
    bucket_id = m.group(1) if m else "unknown"

    data = load_prompt_list(filename, start=0, end=25000)
    data_eos_id = min(len(data), int(data_eos_id))
    new_data: list[dict] = []

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos_id if eos_id is not None else 0)

    for start_idx in tqdm(range(int(data_bos_id), int(data_eos_id), batch_size), desc="trajectory"):
        end_idx = min(start_idx + batch_size, int(data_eos_id))
        batch_indices = list(range(start_idx, end_idx))
        batch_user_texts = [data[idx] for idx in batch_indices]
        rendered_prompts = [build_prompt(tokenizer, user_text, system_prompt) for user_text in batch_user_texts]
        batch_inputs = tokenizer(rendered_prompts, return_tensors="pt", padding=True).to(model.device)

        prompt_token_lists: list[list[int]] = []
        prompt_lens: list[int] = []
        for row_ids, row_mask in zip(batch_inputs["input_ids"], batch_inputs["attention_mask"]):
            true_ids = row_ids[row_mask.bool()].tolist()
            prompt_token_lists.append([int(x) for x in true_ids])
            prompt_lens.append(len(true_ids))

        seed_tokens = prefill_first_correct_tokens_batch(model, prompt_token_lists, pad_id)
        generated_prefixes = [list(x) for x in prompt_token_lists]
        dict_lsts: list[list[dict]] = [[] for _ in batch_indices]
        finished = [False] * len(batch_indices)

        while not all(finished):
            active_indices = [
                i for i in range(len(batch_indices))
                if not finished[i]
                and (len(generated_prefixes[i]) - prompt_lens[i] < max_new_seq_len)
            ]
            if not active_indices:
                break

            active_prefixes = [generated_prefixes[i] for i in active_indices]
            active_seeds = [seed_tokens[i] for i in active_indices]
            block_results = generate_one_block_trajectory_batch(
                model=model,
                prefixes=active_prefixes,
                seed_tokens=active_seeds,
                block_len=n_token_seq_len,
                pad_id=pad_id,
                eos_id=eos_id,
            )

            for local_idx, sample_idx in enumerate(active_indices):
                prefix_before = list(generated_prefixes[sample_idx])
                committed_block = list(block_results[local_idx]["committed_block"])
                generated_prefixes[sample_idx].extend(committed_block)
                seed_tokens[sample_idx] = int(block_results[local_idx]["next_seed"])

                dict_lsts[sample_idx].append(
                    {
                        "diffusion_itr_id": f"itr_{len(dict_lsts[sample_idx])}",
                        "data_id": f"bucket_{bucket_id}_data_{batch_indices[sample_idx]}",
                        "prompt_ids": [prefix_before],
                        "answer_trajectory_ids": block_results[local_idx]["trajectory_states"],
                        "teacher_output_ids": list(generated_prefixes[sample_idx]),
                    }
                )

                stop_hit = bool(block_results[local_idx]["stop_hit"])
                reached_cap = (len(generated_prefixes[sample_idx]) - prompt_lens[sample_idx]) >= max_new_seq_len
                if stop_hit or reached_cap:
                    finished[sample_idx] = True

        for sample_idx, records in enumerate(dict_lsts):
            if not records:
                continue
            best_teacher_output = max(records, key=lambda x: len(x["teacher_output_ids"]))["teacher_output_ids"]
            for item in records:
                item["teacher_output_ids"] = best_teacher_output
                new_data.append(item)

    os.makedirs(save_path, exist_ok=True)
    out_name = (
        f"{Path(filename).stem}_exaone4_greedy_jacobi_len{n_token_seq_len}"
        f"_maxlen{max_new_seq_len}_{data_bos_id}_{data_eos_id}.json"
    )
    out_path = os.path.join(save_path, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False)

    print(json.dumps({"output_path": out_path, "num_records": len(new_data)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_token_seq_len", type=int, default=64)
    parser.add_argument("--max_new_seq_len", type=int, default=1024)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_bos_id", type=int, default=0)
    parser.add_argument("--data_eos_id", type=int, default=40)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="flex_attention")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    main(
        filename=args.filename,
        model=model,
        tokenizer=tokenizer,
        n_token_seq_len=args.n_token_seq_len,
        max_new_seq_len=args.max_new_seq_len,
        data_bos_id=args.data_bos_id,
        data_eos_id=args.data_eos_id,
        save_path=args.save_path,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
    )
