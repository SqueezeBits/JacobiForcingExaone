#!/usr/bin/env python3
"""Create deterministic 2-step split files from bucketed OpenCodeInstruct records."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def load_bucket_records(input_path: Path) -> tuple[List[Dict], List[str]]:
    bucket_files = sorted(input_path.glob("bucket_*.json"))
    if not bucket_files:
        raise FileNotFoundError(f"No bucket_*.json files found under {input_path}")

    all_records: List[Dict] = []
    for bucket_file in bucket_files:
        with bucket_file.open("r", encoding="utf-8") as fin:
            bucket_records = json.load(fin)
        if not isinstance(bucket_records, list):
            raise ValueError(f"{bucket_file} does not contain a top-level JSON list")
        for record in bucket_records:
            if not isinstance(record, dict):
                raise ValueError(f"{bucket_file} contains a non-dict record")
            record = dict(record)
            record["source_bucket_file"] = bucket_file.name
            all_records.append(record)
    return all_records, [bucket_file.name for bucket_file in bucket_files]


def summarize_lengths(records: List[Dict]) -> Dict[str, float]:
    if not records:
        return {"count": 0, "min_tokens": 0, "max_tokens": 0, "avg_tokens": 0.0}
    lengths = [int(record["n_tokens"]) for record in records]
    return {
        "count": len(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "avg_tokens": round(sum(lengths) / len(lengths), 4),
    }


def assign_data_ids(records: List[Dict], split_name: str) -> List[Dict]:
    assigned: List[Dict] = []
    for idx, record in enumerate(records):
        updated = dict(record)
        updated["split_name"] = split_name
        updated["data_id"] = f"data_{idx}"
        assigned.append(updated)
    return assigned


def main():
    parser = argparse.ArgumentParser(description="Create split_a/split_b for 2-step Solar data prep")
    parser.add_argument("--input_path", required=True, help="Directory containing bucket_*.json files")
    parser.add_argument("--output_path", required=True, help="Directory for split files and manifest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic shuffle")
    parser.add_argument("--total_samples", type=int, default=40000, help="Total records to keep before 50:50 split")
    parser.add_argument("--split_a_name", default="split_a", help="Name for first split")
    parser.add_argument("--split_b_name", default="split_b", help="Name for second split")
    parser.add_argument("--chat_template_mode", default="solar", help="Metadata only")
    parser.add_argument("--tokenizer_path", required=True, help="Tokenizer used for bucketing metadata")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    all_records, bucket_files = load_bucket_records(input_path)
    if len(all_records) < args.total_samples:
        raise ValueError(
            f"Requested total_samples={args.total_samples}, but only {len(all_records)} records are available"
        )

    rng = random.Random(args.seed)
    shuffled_records = list(all_records)
    rng.shuffle(shuffled_records)

    selected_records = shuffled_records[: args.total_samples]
    split_boundary_index = len(selected_records) // 2
    split_a_records = assign_data_ids(selected_records[:split_boundary_index], args.split_a_name)
    split_b_records = assign_data_ids(selected_records[split_boundary_index:], args.split_b_name)

    split_a_path = output_path / f"{args.split_a_name}.json"
    split_b_path = output_path / f"{args.split_b_name}.json"
    manifest_path = output_path / "split_manifest.json"

    with split_a_path.open("w", encoding="utf-8") as fout:
        json.dump(split_a_records, fout, ensure_ascii=False, indent=2)
    with split_b_path.open("w", encoding="utf-8") as fout:
        json.dump(split_b_records, fout, ensure_ascii=False, indent=2)

    manifest = {
        "dataset_name": "OpenCodeInstruct",
        "tokenizer_path": args.tokenizer_path,
        "chat_template_mode": args.chat_template_mode,
        "seed": args.seed,
        "available_total_records": len(all_records),
        "selected_total_records": len(selected_records),
        "selection_boundary": len(selected_records),
        "split_boundary_index": split_boundary_index,
        "split_a_name": args.split_a_name,
        "split_b_name": args.split_b_name,
        "split_a_count": len(split_a_records),
        "split_b_count": len(split_b_records),
        "source_bucket_files": bucket_files,
        "split_a_stats": summarize_lengths(split_a_records),
        "split_b_stats": summarize_lengths(split_b_records),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with manifest_path.open("w", encoding="utf-8") as fout:
        json.dump(manifest, fout, ensure_ascii=False, indent=2)

    print(f"Wrote {split_a_path}")
    print(f"Wrote {split_b_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
