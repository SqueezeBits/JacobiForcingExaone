#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate JacobiForcing greedy trajectories with vLLM offline inference.

This script avoids HF model-cache surgery by scoring each Jacobi draft block via
vLLM.  The fast path asks vLLM for prompt_logprobs on ``context + draft_block``
while keeping automatic prefix caching (APC) enabled.  vLLM normally disables
prefix-cache reads for prompt_logprobs because cached prompt positions may not
produce logprobs; therefore this script validates every required draft-block
position and falls back when a cached/boundary position is missing.

Fallback choices:
  * non_apc_prompt_logprobs: one full ``context + draft`` scoring request with
    prefix-cache reads disabled per request.
  * apc_multi_prefix: ``n`` one-token greedy generation requests for
    ``context + draft[:1]``, ..., ``context + draft[:n]`` with APC enabled.

The auto fallback mode benchmarks both choices on the first fallback blocks and
uses the faster valid method afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoTokenizer

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

    # greedy_all[:len(block)] are the greedy tokens for each block position.
    # greedy_all[-1] is the next token after the full block.
    greedy_all: list[int]
    method: str
    num_cached_tokens: int = 0
    elapsed_s: float = 0.0


@dataclass
class FallbackStats:
    method_times: dict[str, float] = field(
        default_factory=lambda: {"non_apc_prompt_logprobs": 0.0, "apc_multi_prefix": 0.0}
    )
    method_counts: dict[str, int] = field(
        default_factory=lambda: {"non_apc_prompt_logprobs": 0, "apc_multi_prefix": 0}
    )
    chosen_method: str | None = None
    calibration_attempts: int = 0
    mismatches: int = 0


def maybe_prepend_vllm_repo(vllm_repo: str | None) -> None:
    if not vllm_repo:
        return
    repo_path = Path(vllm_repo).expanduser().resolve()
    if repo_path.exists():
        sys.path.insert(0, str(repo_path))


def import_vllm(vllm_repo: str | None):
    maybe_prepend_vllm_repo(vllm_repo)
    from vllm import LLM, SamplingParams  # type: ignore
    from vllm.inputs import TokensPrompt  # type: ignore

    return LLM, SamplingParams, TokensPrompt


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


def top1_token_from_logprobs(logprobs_for_position: Any) -> int | None:
    """Return the rank-1 token id from vLLM's per-position prompt_logprobs."""
    if not logprobs_for_position:
        return None
    # Standard non-flat prompt_logprobs shape: dict[token_id, Logprob].
    if isinstance(logprobs_for_position, dict):
        for token_id, logprob in logprobs_for_position.items():
            if getattr(logprob, "rank", None) == 1:
                return int(token_id)
        return None
    return None


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
        fallback_mode: str = "auto",
        fallback_calibration_blocks: int = 2,
    ) -> None:
        self.llm = llm
        self.SamplingParams = SamplingParams
        self.TokensPrompt = TokensPrompt
        self.n_token_seq_len = n_token_seq_len
        self.fallback_mode = fallback_mode
        self.fallback_calibration_blocks = fallback_calibration_blocks
        self.stats = FallbackStats()

        self.greedy_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=0,
            detokenize=False,
        )
        self.apc_prompt_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
            logprobs=0,
            detokenize=False,
            skip_reading_prefix_cache=False,
        )
        self.non_apc_prompt_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
            logprobs=0,
            detokenize=False,
            skip_reading_prefix_cache=True,
        )

    def tokens_prompt(self, token_ids: Sequence[int]) -> Any:
        return self.TokensPrompt(prompt_token_ids=list(map(int, token_ids)))

    def score_block_prompt_logprobs(
        self,
        context_ids: Sequence[int],
        block_ids: Sequence[int],
        *,
        skip_reading_prefix_cache: bool,
        method: str,
    ) -> BlockScore:
        full_ids = list(context_ids) + list(block_ids)
        params = (
            self.non_apc_prompt_logprobs_params
            if skip_reading_prefix_cache
            else self.apc_prompt_logprobs_params
        )
        start = time.perf_counter()
        outputs = self.llm.generate(
            [self.tokens_prompt(full_ids)],
            sampling_params=params,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - start
        output = outputs[0]
        prompt_logprobs = output.prompt_logprobs
        ctx_len = len(context_ids)
        block_len = len(block_ids)
        if block_len < 1:
            raise ValueError("block_ids must contain at least one token")
        if prompt_logprobs is None:
            raise RuntimeError(f"{method}: prompt_logprobs is None")
        if len(prompt_logprobs) < len(full_ids):
            raise RuntimeError(
                f"{method}: prompt_logprobs length {len(prompt_logprobs)} < prompt length {len(full_ids)}; "
                f"num_cached_tokens={get_num_cached_tokens(output)}"
            )

        greedy_all: list[int] = []
        # Position ctx_len + i stores the distribution for token full_ids[ctx_len + i]
        # conditioned on previous tokens. Read every block position so the same
        # scoring path can bootstrap the first greedy token as well.
        for pos in range(ctx_len, ctx_len + block_len):
            token_id = top1_token_from_logprobs(prompt_logprobs[pos])
            if token_id is None:
                raise RuntimeError(
                    f"{method}: missing rank-1 prompt logprob at pos={pos}; "
                    f"ctx_len={ctx_len} block_len={block_len} num_cached_tokens={get_num_cached_tokens(output)}"
                )
            greedy_all.append(token_id)

        next_token = first_output_token(output)
        if next_token is None:
            raise RuntimeError(f"{method}: missing generated next token")
        greedy_all.append(next_token)
        return BlockScore(
            greedy_all=greedy_all,
            method=method,
            num_cached_tokens=get_num_cached_tokens(output),
            elapsed_s=elapsed,
        )

    def score_block_apc_prompt_logprobs(self, context_ids: Sequence[int], block_ids: Sequence[int]) -> BlockScore:
        return self.score_block_prompt_logprobs(
            context_ids,
            block_ids,
            skip_reading_prefix_cache=False,
            method="apc_prompt_logprobs",
        )

    def score_block_non_apc_prompt_logprobs(self, context_ids: Sequence[int], block_ids: Sequence[int]) -> BlockScore:
        return self.score_block_prompt_logprobs(
            context_ids,
            block_ids,
            skip_reading_prefix_cache=True,
            method="non_apc_prompt_logprobs",
        )

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

    def _record_fallback_result(self, score: BlockScore) -> None:
        if score.method in self.stats.method_times:
            self.stats.method_times[score.method] += score.elapsed_s
            self.stats.method_counts[score.method] += 1

    def _choose_from_calibration(self) -> str | None:
        if self.stats.calibration_attempts < self.fallback_calibration_blocks:
            return None
        averages: dict[str, float] = {}
        for method, total_time in self.stats.method_times.items():
            count = self.stats.method_counts[method]
            if count > 0:
                averages[method] = total_time / count
        if not averages:
            return None
        return min(averages, key=averages.get)

    def score_block_fallback(self, context_ids: Sequence[int], block_ids: Sequence[int]) -> BlockScore:
        if self.fallback_mode == "non_apc_prompt_logprobs":
            return self.score_block_non_apc_prompt_logprobs(context_ids, block_ids)
        if self.fallback_mode == "apc_multi_prefix":
            return self.score_block_apc_multi_prefix(context_ids, block_ids)
        if self.fallback_mode != "auto":
            raise ValueError(f"Unknown fallback_mode={self.fallback_mode}")

        if self.stats.chosen_method == "non_apc_prompt_logprobs":
            return self.score_block_non_apc_prompt_logprobs(context_ids, block_ids)
        if self.stats.chosen_method == "apc_multi_prefix":
            return self.score_block_apc_multi_prefix(context_ids, block_ids)

        non_apc_score: BlockScore | None = None
        apc_multi_score: BlockScore | None = None
        non_apc_error: Exception | None = None
        apc_multi_error: Exception | None = None

        try:
            non_apc_score = self.score_block_non_apc_prompt_logprobs(context_ids, block_ids)
            self._record_fallback_result(non_apc_score)
        except Exception as exc:  # noqa: BLE001 - report and try the other fallback.
            non_apc_error = exc

        try:
            apc_multi_score = self.score_block_apc_multi_prefix(context_ids, block_ids)
            self._record_fallback_result(apc_multi_score)
        except Exception as exc:  # noqa: BLE001 - report and try the other fallback.
            apc_multi_error = exc

        if non_apc_score is None and apc_multi_score is None:
            raise RuntimeError(
                "Both fallback modes failed: "
                f"non_apc_prompt_logprobs={non_apc_error!r}; apc_multi_prefix={apc_multi_error!r}"
            )
        if non_apc_score is not None and apc_multi_score is not None:
            if non_apc_score.greedy_all != apc_multi_score.greedy_all:
                self.stats.mismatches += 1
                # Prefer the mathematically direct full-prompt scoring path when
                # the fallback methods disagree; it recomputes prompt logprobs.
                # Also pin future auto fallbacks to this conservative method; a
                # faster multi-prefix path is not useful if it disagrees.
                self.stats.calibration_attempts += 1
                self.stats.chosen_method = "non_apc_prompt_logprobs"
                return non_apc_score

            chosen = non_apc_score if non_apc_score.elapsed_s <= apc_multi_score.elapsed_s else apc_multi_score
            self.stats.calibration_attempts += 1
            self.stats.chosen_method = self._choose_from_calibration()
            return chosen

        chosen = non_apc_score or apc_multi_score
        assert chosen is not None
        self.stats.calibration_attempts += 1
        self.stats.chosen_method = chosen.method
        return chosen

    def score_block(self, context_ids: Sequence[int], block_ids: Sequence[int]) -> tuple[BlockScore, str | None]:
        try:
            return self.score_block_apc_prompt_logprobs(context_ids, block_ids), None
        except Exception as exc:  # noqa: BLE001 - fallback is intentional here.
            fallback = self.score_block_fallback(context_ids, block_ids)
            return fallback, str(exc)

    def bootstrap_first_token(self, context_ids: Sequence[int], draft_token: int) -> tuple[int, dict[str, Any]]:
        score, fallback_reason = self.score_block(context_ids, [int(draft_token)])
        if not score.greedy_all:
            raise RuntimeError("bootstrap_first_token: missing greedy token for first block position")
        return int(score.greedy_all[0]), {
            "method": score.method,
            "fallback_reason": fallback_reason,
            "num_cached_tokens": score.num_cached_tokens,
            "elapsed_s": score.elapsed_s,
            "bootstrap": True,
        }


def run_jacobi_block(
    scorer: VLLMJacobiScorer,
    context_ids: list[int],
    n_token_seq_len: int,
    tokenizer,
    rng: random.Random,
) -> tuple[int, list[list[int]], list[dict[str, Any]]]:
    seed_token = context_ids[-1]
    first_correct_token, bootstrap_diag = scorer.bootstrap_first_token(context_ids, seed_token)
    block = [int(first_correct_token)] + sample_draft_tokens(context_ids, n_token_seq_len, rng)
    accepted_n_gram = list(block)
    answer_trajectory_ids = [list(block)]
    total_accepted = 0
    next_token = int(first_correct_token)
    diagnostics: list[dict[str, Any]] = [bootstrap_diag]
    score_context = list(context_ids)

    while total_accepted < n_token_seq_len:
        # `score_context` tracks the original context plus the prefix accepted
        # inside this Jacobi block. This mirrors the HF implementation's
        # past_key_values after cache cropping: subsequent refinement passes are
        # conditioned on accepted tokens, not only on the pre-block context.
        score, fallback_reason = scorer.score_block(score_context, block)
        greedy_all = score.greedy_all
        greedy_tokens = greedy_all[:-1]
        next_after_block = greedy_all[-1]
        if len(greedy_tokens) != len(block):
            raise RuntimeError(
                f"greedy token count mismatch: got {len(greedy_tokens)} expected {len(block)}"
            )

        mismatch_idx: int | None = None
        for idx, (draft_token, greedy_token) in enumerate(zip(block, greedy_tokens)):
            if int(draft_token) != int(greedy_token):
                mismatch_idx = idx
                break
        num_accepted_raw = (len(block) if mismatch_idx is None else mismatch_idx)

        eos_id = tokenizer.eos_token_id
        num_accepted = num_accepted_raw
        if eos_id is not None:
            for idx, token in enumerate(block[:num_accepted_raw]):
                if int(token) == int(eos_id):
                    num_accepted = idx + 1
                    break

        if num_accepted > 0:
            accepted_n_gram[total_accepted : total_accepted + num_accepted] = block[:num_accepted]
        total_accepted += num_accepted
        diagnostics.append(
            {
                "method": score.method,
                "fallback_reason": fallback_reason,
                "num_cached_tokens": score.num_cached_tokens,
                "elapsed_s": score.elapsed_s,
                "num_accepted_raw": num_accepted_raw,
                "num_accepted": num_accepted,
            }
        )

        if eos_id is not None and any(int(token) == int(eos_id) for token in block[:num_accepted]):
            next_token = int(eos_id)
            return next_token, answer_trajectory_ids, diagnostics

        has_rejected = num_accepted_raw < len(block)
        if has_rejected:
            next_token = int(greedy_tokens[num_accepted_raw])
            if eos_id is not None and next_token == int(eos_id):
                if total_accepted < len(accepted_n_gram):
                    accepted_n_gram[total_accepted : total_accepted + 1] = [next_token]
                total_accepted += 1
                answer_trajectory_ids.append(accepted_n_gram[: min(total_accepted, n_token_seq_len)])
                return next_token, answer_trajectory_ids, diagnostics

            remaining_greedy = [int(tok) for tok in greedy_tokens[num_accepted_raw + 1 :]]
            block = [next_token] + remaining_greedy
            score_context = list(context_ids) + accepted_n_gram[:total_accepted]
            answer_trajectory_ids.append(accepted_n_gram[:total_accepted] + block)
            continue

        next_token = int(next_after_block)
        if total_accepted < len(accepted_n_gram):
            accepted_n_gram[total_accepted : total_accepted + 1] = [next_token]
        if eos_id is not None and next_token == int(eos_id):
            total_accepted += 1
            answer_trajectory_ids.append(accepted_n_gram[: min(total_accepted, n_token_seq_len)])
            return next_token, answer_trajectory_ids, diagnostics

        total_accepted += 1
        answer_trajectory_ids.append(accepted_n_gram[: min(total_accepted, n_token_seq_len)])

    return next_token, answer_trajectory_ids, diagnostics


def process_record(
    record: dict[str, Any],
    record_idx: int,
    scorer: VLLMJacobiScorer,
    tokenizer,
    n_token_seq_len: int,
    max_new_seq_len: int,
    chat_template_mode: str,
    seed: int,
    include_diagnostics: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_id = str(record.get("data_id", f"data_{record_idx}"))
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return [], [{"data_id": data_id, "reason": "missing_prompt"}]

    skip_records: list[dict[str, Any]] = []
    try:
        input_ids = encode_prompt(tokenizer, prompt, chat_template_mode)
    except Exception as exc:  # noqa: BLE001
        return [], [{"data_id": data_id, "reason": "tokenization_failed", "error": repr(exc)}]

    if not input_ids:
        return [], [{"data_id": data_id, "reason": "empty_prompt_after_tokenization"}]

    generated_ids = list(input_ids)
    iterations = 0
    per_iteration_records: list[dict[str, Any]] = []
    rng = random.Random(seed + record_idx)

    while True:
        generated_part = generated_ids[len(input_ids) :]
        if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) in generated_part:
            break
        if iterations * n_token_seq_len >= max_new_seq_len:
            break

        prompt_ids_for_record = list(generated_ids)
        try:
            next_token, answer_trajectory_ids, diagnostics = run_jacobi_block(
                scorer=scorer,
                context_ids=generated_ids,
                n_token_seq_len=n_token_seq_len,
                tokenizer=tokenizer,
                rng=rng,
            )
        except Exception as exc:  # noqa: BLE001
            append_skip(
                skip_records,
                data_id,
                "jacobi_block_failed",
                diffusion_itr_id=f"itr_{iterations}",
                error=repr(exc),
            )
            break

        if not answer_trajectory_ids:
            append_skip(skip_records, data_id, "empty_answer_trajectory", diffusion_itr_id=f"itr_{iterations}")
            break

        final_block = answer_trajectory_ids[-1]
        final_block_len = len(final_block)
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

        generated_ids.extend(final_block)
        row: dict[str, Any] = {
            "diffusion_itr_id": f"itr_{iterations}",
            "data_id": data_id,
            "prompt_ids": [prompt_ids_for_record],
            "answer_trajectory_ids": answer_trajectory_ids,
        }
        if include_diagnostics:
            row["vllm_diagnostics"] = diagnostics
        per_iteration_records.append(row)
        iterations += 1

    if not per_iteration_records:
        if not skip_records:
            append_skip(skip_records, data_id, "no_valid_iterations")
        return [], skip_records

    teacher_output_ids = list(generated_ids)
    for iteration_record in per_iteration_records:
        iteration_record["teacher_output_ids"] = teacher_output_ids

    return per_iteration_records, skip_records


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
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
    }
    if args.max_model_len > 0:
        kwargs["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    return kwargs


def main(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)
    LLM, SamplingParams, TokensPrompt = import_vllm(args.vllm_repo)

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
        fallback_mode=args.fallback_mode,
        fallback_calibration_blocks=args.fallback_calibration_blocks,
    )

    records = load_records(
        args.filename,
        start=int(args.data_bos_id),
        end=None if int(args.data_eos_id) < 0 else int(args.data_eos_id),
    )

    generated_records: list[dict[str, Any]] = []
    skip_records: list[dict[str, Any]] = []
    for local_idx, record in enumerate(tqdm(records, desc="Generating vLLM trajectories", total=len(records))):
        sample_records, sample_skips = process_record(
            record=record,
            record_idx=local_idx + int(args.data_bos_id),
            scorer=scorer,
            tokenizer=tokenizer,
            n_token_seq_len=args.n_token_seq_len,
            max_new_seq_len=args.max_new_seq_len,
            chat_template_mode=args.chat_template_mode,
            seed=args.seed,
            include_diagnostics=args.include_diagnostics,
        )
        generated_records.extend(sample_records)
        skip_records.extend(sample_skips)

    os.makedirs(args.save_path, exist_ok=True)
    stem = Path(args.filename).stem
    range_suffix = f"{int(args.data_bos_id)}_{int(args.data_eos_id)}"
    model_family = "solar"
    output_file = os.path.join(
        args.save_path,
        f"{stem}_{args.chat_template_mode}_{model_family}_vllm_greedy_jacobi_len{args.n_token_seq_len}_{range_suffix}.json",
    )
    skip_file = os.path.join(
        args.save_path,
        f"{stem}_{args.chat_template_mode}_{model_family}_vllm_greedy_jacobi_len{args.n_token_seq_len}_{range_suffix}_skipped.jsonl",
    )
    stats_file = os.path.join(
        args.save_path,
        f"{stem}_{args.chat_template_mode}_{model_family}_vllm_greedy_jacobi_len{args.n_token_seq_len}_{range_suffix}_stats.json",
    )

    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(generated_records, fout, ensure_ascii=False)
    with open(skip_file, "w", encoding="utf-8") as fout:
        for skip_record in skip_records:
            fout.write(json.dumps(skip_record, ensure_ascii=False) + "\n")
    with open(stats_file, "w", encoding="utf-8") as fout:
        json.dump(
            {
                "fallback": {
                    "mode": args.fallback_mode,
                    "chosen_method": scorer.stats.chosen_method,
                    "method_times": scorer.stats.method_times,
                    "method_counts": scorer.stats.method_counts,
                    "calibration_attempts": scorer.stats.calibration_attempts,
                    "mismatches": scorer.stats.mismatches,
                },
                "num_generated_records": len(generated_records),
                "num_skipped_records": len(skip_records),
            },
            fout,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {len(generated_records)} trajectory records to {output_file}")
    print(f"Wrote {len(skip_records)} skip records to {skip_file}")
    print(f"Wrote fallback stats to {stats_file}")


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
        help="Use processed_logprobs by default so prompt_logprobs match global Solar logits processors.",
    )
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--vllm_repo", default=os.environ.get("VLLM_REPO"))
    parser.add_argument(
        "--logits_processors",
        action="append",
        default=[
            "vllm.model_executor.models.parallel_tool_call_logits_processor:ParallelToolCallLogitsProcessor",
            "vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor",
        ],
        help="Global vLLM logits processor class path. Repeat to pass multiple; defaults match Solar-Open HF vLLM guide.",
    )
    parser.add_argument(
        "--fallback_mode",
        choices=["auto", "non_apc_prompt_logprobs", "apc_multi_prefix"],
        default="auto",
    )
    parser.add_argument("--fallback_calibration_blocks", type=int, default=2)
    parser.add_argument("--include_diagnostics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
