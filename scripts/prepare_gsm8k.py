#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PYDEPS = Path("/data/dong/explore/pydeps")
if LOCAL_PYDEPS.exists():
    sys.path.insert(0, str(LOCAL_PYDEPS))

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "pyarrow is required to read the downloaded GSM8K parquet files. "
        "Install pyarrow or place it in /data/dong/explore/pydeps."
    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert downloaded GSM8K parquet files to JSONL.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/gsm8k/raw")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data/gsm8k")
    parser.add_argument("--subset", choices=["main", "socratic"], default="main")
    parser.add_argument("--train-limit", type=int, default=None)
    args = parser.parse_args()

    subset_dir = args.raw_dir / args.subset
    train_rows = read_parquet(subset_dir / "train-00000-of-00001.parquet")
    test_rows = read_parquet(subset_dir / "test-00000-of-00001.parquet")
    if args.train_limit is not None:
        train_rows = train_rows[: args.train_limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / f"{args.subset}_train.jsonl", normalize_rows(train_rows, "train"))
    write_jsonl(args.out_dir / f"{args.subset}_test.jsonl", normalize_rows(test_rows, "test"))
    write_jsonl(args.out_dir / f"{args.subset}_train_sft.jsonl", to_sft_rows(train_rows, "train"))
    write_jsonl(args.out_dir / f"{args.subset}_test_sft.jsonl", to_sft_rows(test_rows, "test"))

    manifest = {
        "dataset": "openai/gsm8k",
        "subset": args.subset,
        "raw_dir": str(args.raw_dir),
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "formats": {
            "raw_jsonl": ["question", "answer", "split", "id"],
            "sft_jsonl": ["messages", "question", "answer", "split", "id"],
        },
    }
    write_json(args.out_dir / f"{args.subset}_manifest.json", manifest)
    print(f"Wrote {len(train_rows)} train and {len(test_rows)} test examples to {args.out_dir}")


def read_parquet(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def normalize_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"gsm8k-{split}-{idx:05d}",
            "split": split,
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "final_answer": extract_final_answer(str(row["answer"])),
        }
        for idx, row in enumerate(rows)
    ]


def to_sft_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"gsm8k-{split}-{idx:05d}",
            "split": split,
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "final_answer": extract_final_answer(str(row["answer"])),
            "messages": [
                {
                    "role": "user",
                    "content": "Solve the following math problem step by step.\n\n"
                    + str(row["question"]).strip(),
                },
                {
                    "role": "assistant",
                    "content": str(row["answer"]).strip(),
                },
            ],
        }
        for idx, row in enumerate(rows)
    ]


def extract_final_answer(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        return ""
    return answer.split(marker, 1)[1].strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
