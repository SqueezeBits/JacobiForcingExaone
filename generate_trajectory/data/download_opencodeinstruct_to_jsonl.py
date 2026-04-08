#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download nvidia/OpenCodeInstruct and export it as JSONL shards."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where JSONL shards will be written.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="HF split to download. Default: train",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional max number of samples to export. 0 means all.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100000,
        help="Number of rows per output JSONL shard.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode to reduce local memory usage.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "nvidia/OpenCodeInstruct",
        split=args.split,
        streaming=args.streaming,
    )

    shard_idx = 0
    row_idx = 0
    shard_count = 0
    current_path = output_dir / f"opencodeinstruct_{args.split}_{shard_idx:05d}.jsonl"
    fout = current_path.open("w", encoding="utf-8")

    def rotate_file():
        nonlocal shard_idx, shard_count, fout, current_path
        fout.close()
        shard_idx += 1
        shard_count = 0
        current_path = output_dir / f"opencodeinstruct_{args.split}_{shard_idx:05d}.jsonl"
        fout = current_path.open("w", encoding="utf-8")

    try:
        for sample in dataset:
            if args.max_samples and row_idx >= args.max_samples:
                break

            # Keep the original row as-is; bucketing script only requires input/output.
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            row_idx += 1
            shard_count += 1

            if shard_count >= args.shard_size:
                rotate_file()
    finally:
        fout.close()

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows_written": row_idx,
                "last_shard_index": shard_idx,
                "streaming": args.streaming,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
