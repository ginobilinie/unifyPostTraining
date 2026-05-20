#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from unify_post_training.gsm8k import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert TruthfulQA multiple-choice data.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/truthfulqa"))
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=args.split)
    rows = []
    for idx, row in enumerate(dataset):
        targets = row["mc1_targets"]
        choices = list(targets["choices"])
        labels = [int(label) for label in targets["labels"]]
        answer_index = labels.index(1) if 1 in labels else 0
        rows.append(
            {
                "id": f"truthfulqa-{args.split}-{idx:05d}",
                "split": args.split,
                "question": row["question"],
                "choices": choices,
                "labels": labels,
                "answer_index": answer_index,
                "answer": choices[answer_index],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.split}_mc1.jsonl"
    write_jsonl(out, rows)
    write_json(
        args.out_dir / "manifest.json",
        {
            "dataset": "truthfulqa/truthful_qa",
            "config": "multiple_choice",
            "split": args.split,
            "examples": len(rows),
            "format": ["id", "question", "choices", "labels", "answer_index", "answer"],
        },
    )
    print(f"Wrote {len(rows)} TruthfulQA examples to {out}")


if __name__ == "__main__":
    main()
