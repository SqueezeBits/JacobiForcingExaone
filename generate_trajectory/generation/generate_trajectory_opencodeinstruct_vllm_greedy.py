#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate JacobiForcing greedy trajectories with vLLM offline inference.

The production path keeps APC enabled and uses ``apc_multi_prefix``:
``n`` one-token greedy generation requests for ``context + draft[:1]``, ...,
``context + draft[:n]`` with APC enabled.
"""



from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoTokenizer


from vllm import LLM, SamplingParams  # type: ignore
from vllm.inputs import TokensPrompt  # type: ignore

# Suppress a noisy third-party deprecation warning emitted by nvidia_cutlass_dsl
# during Solar FP8 model load.
warnings.filterwarnings(
    "ignore",
    message=r"Use explicit `struct\.scalar\.ptr` for pointer instead\.",
    category=DeprecationWarning,
)


@dataclass
class BlockScore:
    """Greedy predictions for a candidate Jacobi block."""

    # greedy_all[i] is the greedy token after context + block[: i + 1].
    # Thus greedy_all[:-1] is compared with block[1:], and greedy_all[-1]
    # is the next token after the full block.
    greedy_all: list[int]
    method: str
    num_cached_tokens: int = 0
    elapsed_s: float = 0.0


@dataclass
class PromptGenerationState:
    record_idx: int
    data_id: str
    input_ids: list[int]
    generated_ids: list[int]
    rng: random.Random
    iterations: int = 0
    per_iteration_records: list[dict[str, Any]] = field(default_factory=list)
    skip_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class JacobiBlockState:
    prompt_state: PromptGenerationState
    prompt_ids_for_record: list[int]
    block: list[int]
    accepted_n_gram: list[int]
    answer_trajectory_ids: list[list[int]]
    diagnostics: list[dict[str, Any]]
    score_context: list[int]
    total_accepted: int = 0
    next_token: int = 0



def resolve_local_hf_snapshot(model_ref: str) -> str:
    """Prefer an already-cached local snapshot for HF repo ids."""
    if not model_ref or Path(model_ref).exists():
        return model_ref
    if model_ref.count("/") != 1:
        return model_ref
    try:
        return snapshot_download(model_ref, local_files_only=True)
    except Exception:  # noqa: BLE001 - keep remote resolution as fallback.
        return model_ref


def set_random_seed(seed: int) -> None:
    random.seed(seed)


def load_records(filename: str, start: int = 0, end: int | None = None) -> list[dict[str, Any]]:
    with open(filename, "r", encoding="utf-8") as fin:
        payload = json.load(fin)
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level JSON list in {filename}")

    end = len(payload) if end is None else min(end, len(payload))
    selected = payload[start:end]
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(selected, start=start):
        if isinstance(item, str):
            normalized.append({"data_id": f"data_{idx}", "prompt": item})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Record {idx} in {filename} is not a dict")
        normalized.append(item)
    return normalized


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


def encode_prompt(tokenizer, prompt: str, chat_template_mode: str) -> list[int]:
    prompt_text = build_generation_prompt(tokenizer, prompt, chat_template_mode)
    return tokenizer(prompt_text, add_special_tokens=False).input_ids


def append_skip(skip_records: list[dict[str, Any]], data_id: str, reason: str, **extra: Any) -> None:
    record = {"data_id": data_id, "reason": reason}
    record.update(extra)
    skip_records.append(record)


def sample_draft_tokens(generated_ids: list[int], n_token_seq_len: int, rng: random.Random) -> list[int]:
    if not generated_ids:
        raise ValueError("Cannot sample draft tokens from an empty generated prefix")
    return rng.choices(generated_ids, k=n_token_seq_len - 1)


def first_output_token(output: Any) -> int | None:
    if not output.outputs:
        return None
    token_ids = list(output.outputs[0].token_ids)
    if not token_ids:
        return None
    return int(token_ids[0])


def get_num_cached_tokens(output: Any) -> int:
    value = getattr(output, "num_cached_tokens", 0)
    return int(value or 0)


class VLLMJacobiScorer:
    def __init__(
        self,
        llm: Any,
        SamplingParams: Any,
        TokensPrompt: Any,
        *,
        n_token_seq_len: int,
        vocab_size: int | None = None,
    ) -> None:
        self.llm = llm
        self.SamplingParams = SamplingParams
        self.TokensPrompt = TokensPrompt
        self.n_token_seq_len = n_token_seq_len
        self.vocab_size = vocab_size

        self.greedy_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=0,
            detokenize=False,
        )
    def tokens_prompt(self, token_ids: Sequence[int]) -> Any:
        return self.TokensPrompt(prompt_token_ids=list(map(int, token_ids)))

    def prefill_next_tokens(self, contexts: Sequence[Sequence[int]]) -> list[int]:
        outputs = self.llm.generate(
            [self.tokens_prompt(context_ids) for context_ids in contexts],
            sampling_params=self.greedy_params,
            use_tqdm=False,
        )
        next_tokens: list[int] = []
        for output in outputs:
            token = first_output_token(output)
            if token is None:
                raise RuntimeError("prefill_next_tokens: missing generated next token")
            next_tokens.append(int(token))
        return next_tokens

    def score_block_apc_multi_prefix(self, context_ids: Sequence[int], block_ids: Sequence[int]) -> BlockScore:
        prompts = [
            self.tokens_prompt(list(context_ids) + list(block_ids[:prefix_len]))
            for prefix_len in range(1, len(block_ids) + 1)
        ]
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params=self.greedy_params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        greedy_all: list[int] = []
        cached = 0
        for idx, output in enumerate(outputs):
            token = first_output_token(output)
            if token is None:
                raise RuntimeError(f"apc_multi_prefix: missing generated next token for prefix_idx={idx}")
            greedy_all.append(token)
            cached += get_num_cached_tokens(output)
        return BlockScore(
            greedy_all=greedy_all,
            method="apc_multi_prefix",
            num_cached_tokens=cached,
            elapsed_s=elapsed,
        )

    def score_block(self, context_ids: Sequence[int], block_ids: Sequence[int]) -> tuple[BlockScore, str | None]:
        return self.score_block_apc_multi_prefix(context_ids, block_ids), None

    def score_blocks(
        self,
        contexts_and_blocks: Sequence[tuple[Sequence[int], Sequence[int]]],
    ) -> list[tuple[BlockScore, str | None]]:
        return [(self.score_block_apc_multi_prefix(context_ids, block_ids), None) for context_ids, block_ids in contexts_and_blocks]


def initialize_jacobi_block_states(
    prompt_states: Sequence[PromptGenerationState],
    scorer: VLLMJacobiScorer,
    n_token_seq_len: int,
) -> list[JacobiBlockState]:
    first_tokens = scorer.prefill_next_tokens([state.generated_ids for state in prompt_states])
    block_states: list[JacobiBlockState] = []
    for state, first_correct_token in zip(prompt_states, first_tokens):
        block = [int(first_correct_token)] + sample_draft_tokens(state.generated_ids, n_token_seq_len, state.rng)
        block_states.append(
            JacobiBlockState(
                prompt_state=state,
                prompt_ids_for_record=list(state.generated_ids),
                block=block,
                accepted_n_gram=list(block),
                answer_trajectory_ids=[list(block)],
                diagnostics=[
                    {
                        "method": "prefill_greedy",
                        "num_cached_tokens": 0,
                        "elapsed_s": 0.0,
                        "bootstrap": True,
                    }
                ],
                score_context=list(state.generated_ids),
                next_token=int(first_correct_token),
            )
        )
    return block_states


def advance_jacobi_block_state(
    block_state: JacobiBlockState,
    score: BlockScore,
    tokenizer,
    n_token_seq_len: int,
) -> bool:
    greedy_all = score.greedy_all
    greedy_tokens = greedy_all[:-1]
    next_after_block = greedy_all[-1]
    block = block_state.block
    if len(greedy_tokens) != len(block) - 1:
        raise RuntimeError(
            f"greedy token count mismatch: got {len(greedy_tokens)} expected {len(block) - 1}"
        )

    mismatch_idx: int | None = None
    for idx, (draft_token, greedy_token) in enumerate(zip(block[1:], greedy_tokens)):
        if int(draft_token) != int(greedy_token):
            mismatch_idx = idx
            break
    num_accepted_raw = (len(block) if mismatch_idx is None else mismatch_idx + 1)

    eos_id = tokenizer.eos_token_id
    num_accepted = num_accepted_raw
    if eos_id is not None:
        for idx, token in enumerate(block[:num_accepted_raw]):
            if int(token) == int(eos_id):
                num_accepted = idx + 1
                break

    if num_accepted > 0:
        block_state.accepted_n_gram[
            block_state.total_accepted : block_state.total_accepted + num_accepted
        ] = block[:num_accepted]
    block_state.total_accepted += num_accepted
    block_state.diagnostics.append(
        {
            "method": score.method,
            "num_cached_tokens": score.num_cached_tokens,
            "elapsed_s": score.elapsed_s,
            "num_accepted_raw": num_accepted_raw,
            "num_accepted": num_accepted,
        }
    )

    if eos_id is not None and any(int(token) == int(eos_id) for token in block[:num_accepted]):
        block_state.next_token = int(eos_id)
        return True

    has_rejected = num_accepted_raw < len(block)
    if has_rejected:
        block_state.next_token = int(greedy_tokens[num_accepted_raw - 1])
        if eos_id is not None and block_state.next_token == int(eos_id):
            if block_state.total_accepted < len(block_state.accepted_n_gram):
                block_state.accepted_n_gram[
                    block_state.total_accepted : block_state.total_accepted + 1
                ] = [block_state.next_token]
            block_state.total_accepted += 1
            block_state.answer_trajectory_ids.append(
                block_state.accepted_n_gram[: min(block_state.total_accepted, n_token_seq_len)]
            )
            return True

        remaining_greedy = [int(tok) for tok in greedy_tokens[num_accepted_raw:]]
        block_state.block = [block_state.next_token] + remaining_greedy
        block_state.score_context = list(block_state.prompt_state.generated_ids) + block_state.accepted_n_gram[
            : block_state.total_accepted
        ]
        block_state.answer_trajectory_ids.append(
            block_state.accepted_n_gram[:block_state.total_accepted] + block_state.block
        )
        return False

    block_state.next_token = int(next_after_block)
    if block_state.total_accepted < len(block_state.accepted_n_gram):
        block_state.accepted_n_gram[block_state.total_accepted : block_state.total_accepted + 1] = [
            block_state.next_token
        ]
    if eos_id is not None and block_state.next_token == int(eos_id):
        block_state.total_accepted += 1
        block_state.answer_trajectory_ids.append(
            block_state.accepted_n_gram[: min(block_state.total_accepted, n_token_seq_len)]
        )
        return True

    block_state.total_accepted += 1
    block_state.answer_trajectory_ids.append(
        block_state.accepted_n_gram[: min(block_state.total_accepted, n_token_seq_len)]
    )
    return block_state.total_accepted >= n_token_seq_len


def run_jacobi_blocks_batched(
    prompt_states: Sequence[PromptGenerationState],
    scorer: VLLMJacobiScorer,
    n_token_seq_len: int,
    tokenizer,
) -> list[JacobiBlockState]:
    pending = initialize_jacobi_block_states(prompt_states, scorer, n_token_seq_len)
    completed: list[JacobiBlockState] = []
    while pending:
        score_results = scorer.score_blocks(
            [(block_state.score_context, block_state.block) for block_state in pending]
        )
        next_pending: list[JacobiBlockState] = []
        for block_state, (score, _) in zip(pending, score_results):
            done = advance_jacobi_block_state(
                block_state=block_state,
                score=score,
                tokenizer=tokenizer,
                n_token_seq_len=n_token_seq_len,
            )
            if done:
                completed.append(block_state)
            else:
                next_pending.append(block_state)
        pending = next_pending
    return completed


def initialize_prompt_state(
    record: dict[str, Any],
    record_idx: int,
    tokenizer,
    chat_template_mode: str,
    seed: int,
) -> tuple[PromptGenerationState | None, list[dict[str, Any]]]:
    data_id = str(record.get("data_id", f"data_{record_idx}"))
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, [{"data_id": data_id, "reason": "missing_prompt"}]

    try:
        input_ids = encode_prompt(tokenizer, prompt, chat_template_mode)
    except Exception as exc:  # noqa: BLE001
        return None, [{"data_id": data_id, "reason": "tokenization_failed", "error": repr(exc)}]

    if not input_ids:
        return None, [{"data_id": data_id, "reason": "empty_prompt_after_tokenization"}]

    return (
        PromptGenerationState(
            record_idx=record_idx,
            data_id=data_id,
            input_ids=list(input_ids),
            generated_ids=list(input_ids),
            rng=random.Random(seed + record_idx),
        ),
        [],
    )


def process_records_batched(
    records: Sequence[dict[str, Any]],
    scorer: VLLMJacobiScorer,
    tokenizer,
    n_token_seq_len: int,
    max_new_seq_len: int,
    chat_template_mode: str,
    seed: int,
    data_bos_id: int,
    output_file: str,
    skip_file: str,
    max_active_prompts: int = 0,
) -> tuple[int, int]:
    prompt_states: list[PromptGenerationState] = []
    num_generated_records = 0
    num_skipped_records = 0
    for local_idx, record in enumerate(records):
        prompt_state, init_skips = initialize_prompt_state(
            record=record,
            record_idx=local_idx + data_bos_id,
            tokenizer=tokenizer,
            chat_template_mode=chat_template_mode,
            seed=seed,
        )
        if prompt_state is not None:
            prompt_states.append(prompt_state)
        if init_skips:
            with open(skip_file, "a", encoding="utf-8") as fout:
                for skip_record in init_skips:
                    fout.write(json.dumps(skip_record, ensure_ascii=False) + "\n")
            num_skipped_records += len(init_skips)

    pending_states = list(prompt_states)
    max_active = max_active_prompts if max_active_prompts > 0 else len(pending_states)
    progress = tqdm(total=len(pending_states), desc="Generating vLLM trajectories")
    active_states: list[PromptGenerationState] = []

    while pending_states or active_states:
        while pending_states and len(active_states) < max_active:
            active_states.append(pending_states.pop(0))

        ready_states: list[PromptGenerationState] = []
        next_active_states: list[PromptGenerationState] = []
        for state in active_states:
            generated_part = state.generated_ids[len(state.input_ids) :]
            if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) in generated_part:
                progress.update(1)
                continue
            if state.iterations * n_token_seq_len >= max_new_seq_len:
                progress.update(1)
                continue
            ready_states.append(state)

        if not ready_states:
            active_states = next_active_states
            continue

        try:
            completed_blocks = run_jacobi_blocks_batched(
                prompt_states=ready_states,
                scorer=scorer,
                n_token_seq_len=n_token_seq_len,
                tokenizer=tokenizer,
            )
        except Exception as exc:  # noqa: BLE001
            batch_skip_rows: list[dict[str, Any]] = []
            for state in ready_states:
                record = {"data_id": state.data_id, "reason": "jacobi_block_failed"}
                record.update(
                    {
                        "diffusion_itr_id": f"itr_{state.iterations}",
                        "error": repr(exc),
                    }
                )
                batch_skip_rows.append(record)
                progress.update(1)
            if batch_skip_rows:
                with open(skip_file, "a", encoding="utf-8") as fout:
                    for skip_record in batch_skip_rows:
                        fout.write(json.dumps(skip_record, ensure_ascii=False) + "\n")
                num_skipped_records += len(batch_skip_rows)
            active_states = next_active_states
            continue

        block_by_record_idx = {block_state.prompt_state.record_idx: block_state for block_state in completed_blocks}
        batch_output_rows: list[dict[str, Any]] = []
        batch_skip_rows: list[dict[str, Any]] = []
        for state in ready_states:
            block_state = block_by_record_idx.get(state.record_idx)
            if block_state is None:
                record = {"data_id": state.data_id, "reason": "missing_batched_block_result"}
                record.update({"diffusion_itr_id": f"itr_{state.iterations}"})
                batch_skip_rows.append(record)
                progress.update(1)
                continue

            answer_trajectory_ids = block_state.answer_trajectory_ids
            if not answer_trajectory_ids:
                record = {"data_id": state.data_id, "reason": "empty_answer_trajectory"}
                record.update({"diffusion_itr_id": f"itr_{state.iterations}"})
                batch_skip_rows.append(record)
                progress.update(1)
                continue

            final_block = answer_trajectory_ids[-1]
            final_block_len = len(final_block)
            if final_block_len != n_token_seq_len:
                record = {"data_id": state.data_id, "reason": "short_final_block"}
                record.update(
                    {
                        "diffusion_itr_id": f"itr_{state.iterations}",
                        "final_block_len": final_block_len,
                        "expected_block_len": n_token_seq_len,
                    }
                )
                batch_skip_rows.append(record)
                progress.update(1)
                continue

            state.generated_ids.extend(final_block)
            row: dict[str, Any] = {
                "diffusion_itr_id": f"itr_{state.iterations}",
                "data_id": state.data_id,
                "prompt_ids": [block_state.prompt_ids_for_record],
                "answer_trajectory_ids": answer_trajectory_ids,
                "vllm_diagnostics": block_state.diagnostics,
                "teacher_output_ids": list(state.generated_ids),
            }
            batch_output_rows.append(row)
            state.iterations += 1

            generated_part = state.generated_ids[len(state.input_ids) :]
            if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) in generated_part:
                progress.update(1)
                continue
            if state.iterations * n_token_seq_len >= max_new_seq_len:
                progress.update(1)
                continue
            next_active_states.append(state)

        active_states = next_active_states

        if batch_output_rows:
            with open(output_file, "a", encoding="utf-8") as fout:
                for row in batch_output_rows:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            num_generated_records += len(batch_output_rows)
        if batch_skip_rows:
            with open(skip_file, "a", encoding="utf-8") as fout:
                for skip_record in batch_skip_rows:
                    fout.write(json.dumps(skip_record, ensure_ascii=False) + "\n")
            num_skipped_records += len(batch_skip_rows)

    progress.close()
    return num_generated_records, num_skipped_records


def build_cudagraph_capture_sizes(n_token_seq_len: int, max_active_prompts: int) -> list[int]:
    max_capture_size = max(1, n_token_seq_len * max(1, max_active_prompts))
    return list(range(1, max_capture_size + 1))


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    cudagraph_capture_sizes = build_cudagraph_capture_sizes(
        args.n_token_seq_len,
        args.max_active_prompts,
    )
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "enable_prefix_caching": True,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "seed": args.seed,
        "disable_log_stats": False,
        "logits_processors": args.logits_processors or None,
        "logprobs_mode": args.logprobs_mode,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "compilation_config": {
            "cudagraph_capture_sizes": cudagraph_capture_sizes,
            "max_cudagraph_capture_size": cudagraph_capture_sizes[-1],
        },
    }
    if args.max_model_len > 0:
        kwargs["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    return kwargs


def main(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)

    args.model = resolve_local_hf_snapshot(args.model)
    args.tokenizer_path = resolve_local_hf_snapshot(args.tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = build_llm_kwargs(args)
    print(
        f"Loading vLLM model={args.model} tokenizer={args.tokenizer_path} "
        f"TP={args.tensor_parallel_size} EP={args.enable_expert_parallel} APC=True"
    )
    llm = LLM(**llm_kwargs)
    scorer = VLLMJacobiScorer(
        llm,
        SamplingParams,
        TokensPrompt,
        n_token_seq_len=args.n_token_seq_len,
        vocab_size=len(tokenizer),
    )

    records = load_records(
        args.filename,
        start=int(args.data_bos_id),
        end=None if int(args.data_eos_id) < 0 else int(args.data_eos_id),
    )

    os.makedirs(args.save_path, exist_ok=True)
    stem = Path(args.filename).stem
    range_suffix = f"{int(args.data_bos_id)}_{int(args.data_eos_id)}"
    model_family = "solar"
    output_file = os.path.join(
        args.save_path,
        f"{stem}_{args.chat_template_mode}_{model_family}_vllm_greedy_jacobi_len{args.n_token_seq_len}_{range_suffix}.jsonl",
    )
    skip_file = os.path.join(
        args.save_path,
        f"{stem}_{args.chat_template_mode}_{model_family}_vllm_greedy_jacobi_len{args.n_token_seq_len}_{range_suffix}_skipped.jsonl",
    )
    stats_file = os.path.join(
        args.save_path,
        f"{stem}_{args.chat_template_mode}_{model_family}_vllm_greedy_jacobi_len{args.n_token_seq_len}_{range_suffix}_stats.json",
    )

    open(output_file, "w", encoding="utf-8").close()
    open(skip_file, "w", encoding="utf-8").close()

    num_generated_records, num_skipped_records = process_records_batched(
        records=records,
        scorer=scorer,
        tokenizer=tokenizer,
        n_token_seq_len=args.n_token_seq_len,
        max_new_seq_len=args.max_new_seq_len,
        chat_template_mode=args.chat_template_mode,
        seed=args.seed,
        data_bos_id=int(args.data_bos_id),
        output_file=output_file,
        skip_file=skip_file,
        max_active_prompts=args.max_active_prompts,
    )

    with open(stats_file, "w", encoding="utf-8") as fout:
        json.dump(
            {
                "scoring": {
                    "mode": "apc_multi_prefix",
                },
                "num_generated_records": num_generated_records,
                "num_skipped_records": num_skipped_records,
            },
            fout,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {num_generated_records} trajectory records to {output_file}")
    print(f"Wrote {num_skipped_records} skip records to {skip_file}")
    print(f"Wrote stats to {stats_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_token_seq_len", type=int, default=16)
    parser.add_argument("--max_new_seq_len", type=int, default=1024)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--data_bos_id", default=0)
    parser.add_argument("--data_eos_id", default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat_template_mode", default="solar", choices=["solar", "qwen"])
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--enable_expert_parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_custom_all_reduce", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--logprobs_mode",
        default="processed_logprobs",
        choices=["raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits"],
        help="Use processed_logprobs by default so Solar logits processors stay aligned.",
    )
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--max_num_batched_tokens", type=int, default=16384)
    parser.add_argument("--max_active_prompts", type=int, default=16)
    parser.add_argument(
        "--logits_processors",
        action="append",
        default=[
            "vllm.model_executor.models.parallel_tool_call_logits_processor:ParallelToolCallLogitsProcessor",
            "vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor",
        ],
        help="Global vLLM logits processor class path. Repeat to pass multiple; defaults match Solar-Open HF vLLM guide.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
