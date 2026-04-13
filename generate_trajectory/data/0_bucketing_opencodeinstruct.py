#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bucket OpenCodeInstruct JSONL records by Solar-compatible chat-template length.

Each bucket file is a JSON array of records with enough metadata to support
deterministic downstream split selection and trajectory generation.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER_PATH: Optional[str] = None
TOKENIZER = None
CHAT_TEMPLATE_MODE = "solar"


def init_worker(tokenizer_path: str):
    global TOKENIZER_PATH, TOKENIZER
    TOKENIZER_PATH = tokenizer_path
    if TOKENIZER is None:
        if not TOKENIZER_PATH:
            raise RuntimeError("TOKENIZER_PATH not set in worker.")
        TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)


def build_messages(user_text: str, assistant_text: str, chat_template_mode: str) -> List[Dict[str, str]]:
    if chat_template_mode != "solar":
        raise ValueError(f"Unsupported --chat_template_mode: {chat_template_mode}")
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def tokenize_pair(user_text: str, assistant_text: str) -> Optional[int]:
    global TOKENIZER
    messages = build_messages(user_text, assistant_text, CHAT_TEMPLATE_MODE)
    try:
        ids = TOKENIZER.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        return int(ids.shape[-1])
    except Exception as exc:
        print(f"Tokenization error: {exc}")
        return None


def process_sample(sample_with_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sample = sample_with_meta["sample"]
    source_file = sample_with_meta["source_file"]
    source_index = sample_with_meta["source_index"]

    prompt = sample.get("input")
    response = sample.get("output")
    if not isinstance(prompt, str) or not isinstance(response, str):
        return None

    prompt_clean = " ".join(prompt.split())
    response_clean = response.strip()
    if not prompt_clean or not response_clean:
        return None

    n_tokens = tokenize_pair(prompt_clean, response_clean)
    if n_tokens is None:
        return None

    return {
        "prompt": prompt_clean,
        "response": response_clean,
        "n_tokens": n_tokens,
        "source_file": source_file,
        "source_index": int(source_index),
        "source_record_id": f"{os.path.basename(source_file)}:{int(source_index)}",
    }


def discover_input_source(input_path: str) -> Tuple[str, List[str], List[str]]:
    path = Path(input_path)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return "local_files", [str(path)], []
        if suffix == ".parquet":
            return "local_files", [], [str(path)]
        raise FileNotFoundError(f"Unsupported file type for {input_path}; expected .jsonl or .parquet")
    if not path.is_dir():
        return "dataset_repo", [], []

    jsonl_paths = sorted(glob.glob(os.path.join(input_path, "**", "*.jsonl"), recursive=True))
    parquet_paths = sorted(glob.glob(os.path.join(input_path, "**", "*.parquet"), recursive=True))
    return "local_files", jsonl_paths, parquet_paths


def load_records_from_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping bad JSON line {line_idx + 1} in {os.path.basename(path)}: {exc}")
                continue
            rows.append(
                {
                    "sample": sample,
                    "source_file": path,
                    "source_index": line_idx,
                }
            )
    return rows


def load_records_from_parquet(path: str) -> List[Dict[str, Any]]:
    dataset = load_dataset("parquet", data_files=path, split="train")
    rows: List[Dict[str, Any]] = []
    for row_idx, sample in enumerate(dataset):
        rows.append(
            {
                "sample": dict(sample),
                "source_file": path,
                "source_index": row_idx,
            }
        )
    return rows


def iter_records_from_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping bad JSON line {line_idx + 1} in {os.path.basename(path)}: {exc}")
                continue
            yield {
                "sample": sample,
                "source_file": path,
                "source_index": line_idx,
            }


def iter_records_from_parquet(paths: List[str], source_name: str) -> Iterator[Dict[str, Any]]:
    dataset = load_dataset("parquet", data_files=paths, split="train", streaming=True)
    for row_idx, sample in enumerate(dataset):
        yield {
            "sample": dict(sample),
            "source_file": source_name,
            "source_index": row_idx,
        }


def iter_records_from_dataset_repo(dataset_name: str, split_name: str) -> Iterator[Dict[str, Any]]:
    dataset = load_dataset(dataset_name, split=split_name, streaming=True)
    for row_idx, sample in enumerate(dataset):
        yield {
            "sample": dict(sample),
            "source_file": dataset_name,
            "source_index": row_idx,
        }


def maybe_reservoir_sample(
    records_iter: Iterable[Dict[str, Any]],
    *,
    max_samples: Optional[int],
    sample_seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    if max_samples is None or max_samples <= 0:
        records = list(records_iter)
        return records, len(records)

    rng = random.Random(sample_seed)
    reservoir: List[Dict[str, Any]] = []
    total_seen = 0
    for total_seen, record in enumerate(records_iter, start=1):
        if len(reservoir) < max_samples:
            reservoir.append(record)
            continue
        swap_idx = rng.randint(0, total_seen - 1)
        if swap_idx < max_samples:
            reservoir[swap_idx] = record
    return reservoir, total_seen


def load_all_records(
    input_path: str,
    input_split: str,
    *,
    max_samples: Optional[int],
    sample_seed: int,
) -> List[Dict[str, Any]]:
    source_kind, jsonl_paths, parquet_paths = discover_input_source(input_path)
    if source_kind == "dataset_repo":
        print(f"STEP 0: Streaming dataset repo {input_path} split={input_split}")
        rows, total_seen = maybe_reservoir_sample(
            iter_records_from_dataset_repo(input_path, input_split),
            max_samples=max_samples,
            sample_seed=sample_seed,
        )
        print(f"STEP 1: Selected {len(rows):,} rows from {total_seen:,} streamed dataset rows")
        return rows
    if not jsonl_paths and not parquet_paths:
        raise FileNotFoundError(f"No .jsonl or .parquet files found under {input_path}")

    print(
        f"STEP 0: Streaming {len(jsonl_paths)} jsonl file(s) and {len(parquet_paths)} parquet file(s)."
    )
    iterators: List[Iterable[Dict[str, Any]]] = [iter_records_from_jsonl(path) for path in jsonl_paths]
    if parquet_paths:
        iterators.append(iter_records_from_parquet(parquet_paths, input_path))
    rows, total_seen = maybe_reservoir_sample(
        chain.from_iterable(iterators),
        max_samples=max_samples,
        sample_seed=sample_seed,
    )
    print(f"STEP 1: Selected {len(rows):,} rows from {total_seen:,} streamed local rows")
    return rows


def main(
    input_path: str,
    output_path: str,
    *,
    tokenizer_path: str,
    chat_template_mode: str,
    input_split: str,
    max_samples: Optional[int],
    sample_seed: int,
    bucket_size: int,
    n_workers: int,
):
    global TOKENIZER_PATH, CHAT_TEMPLATE_MODE
    TOKENIZER_PATH = tokenizer_path
    CHAT_TEMPLATE_MODE = chat_template_mode

    os.makedirs(output_path, exist_ok=True)
    samples = load_all_records(
        input_path,
        input_split,
        max_samples=max_samples,
        sample_seed=sample_seed,
    )

    with mp.Pool(n_workers, initializer=init_worker, initargs=(tokenizer_path,)) as pool:
        processed = list(
            tqdm(
                pool.imap(partial(process_sample), samples),
                total=len(samples),
                desc="Tokenising",
            )
        )

    processed = [record for record in processed if record is not None]
    processed.sort(key=lambda record: (record["n_tokens"], record["source_file"], record["source_index"]))

    print(f"STEP 2: Tokenised {len(processed):,} samples")
    print(f"STEP 3: Bucketing {len(processed):,} records with bucket_size={bucket_size}")

    for start in range(0, len(processed), bucket_size):
        bucket_idx = start // bucket_size
        bucket = processed[start : start + bucket_size]
        if not bucket:
            continue

        token_counts = [record["n_tokens"] for record in bucket]
        bucket_meta = {
            "bucket_index": bucket_idx,
            "avg_tokens": int(round(sum(token_counts) / len(token_counts))),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "count": len(bucket),
            "chat_template_mode": chat_template_mode,
            "tokenizer_path": tokenizer_path,
        }

        output_name = (
            f"bucket_{bucket_idx:04d}"
            f"_avg{bucket_meta['avg_tokens']}"
            f"_min{bucket_meta['min_tokens']}"
            f"_max{bucket_meta['max_tokens']}.json"
        )
        output_file = os.path.join(output_path, output_name)
        with open(output_file, "w", encoding="utf-8") as fout:
            json.dump(bucket, fout, ensure_ascii=False, indent=2)

        print(
            f"-- {output_name}"
            f" ({bucket_meta['count']} records, avg={bucket_meta['avg_tokens']},"
            f" min={bucket_meta['min_tokens']}, max={bucket_meta['max_tokens']})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bucket OpenCodeInstruct records by Solar chat-template token length."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Local .jsonl/.parquet path, local directory, or dataset repo id such as nvidia/OpenCodeInstruct",
    )
    parser.add_argument("--input_split", default="train", help="Dataset split when --input_path is a dataset repo id")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Reservoir-sample at most this many records before tokenization/bucketing",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Random seed for reservoir sampling when --max_samples is set",
    )
    parser.add_argument("--output_path", required=True, help="Directory for bucket files")
    parser.add_argument("--tokenizer_path", required=True, help="HF repo or local tokenizer path")
    parser.add_argument(
        "--chat_template_mode",
        default="solar",
        choices=["solar"],
        help="Chat template mode used for token counting",
    )
    parser.add_argument("--bucket_size", type=int, default=25_000, help="Number of records per bucket")
    parser.add_argument("--n_workers", type=int, default=8, help="Tokenisation workers")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    main(
        args.input_path,
        args.output_path,
        tokenizer_path=args.tokenizer_path,
        chat_template_mode=args.chat_template_mode,
        input_split=args.input_split,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
        bucket_size=args.bucket_size,
        n_workers=args.n_workers,
    )
