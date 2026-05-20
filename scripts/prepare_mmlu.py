#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from unify_post_training.gsm8k import write_json, write_jsonl


DEFAULT_SUBJECTS = [
    "abstract_algebra",
    "college_medicine",
    "computer_security",
    "high_school_mathematics",
    "professional_law",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert an MMLU retention slice.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/mmlu"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--limit-per-subject", type=int, default=100)
    args = parser.parse_args()

    subjects = [subject.strip() for subject in args.subjects.split(",") if subject.strip()]
    dataset = load_dataset("cais/mmlu", "all", split=args.split)
    counts: Counter[str] = Counter()
    rows = []
    for row in dataset:
        subject = row["subject"]
        if subject not in subjects:
            continue
        if counts[subject] >= args.limit_per_subject:
            continue
        counts[subject] += 1
        rows.append(
            {
                "id": f"mmlu-{args.split}-{subject}-{counts[subject] - 1:05d}",
                "split": args.split,
                "subject": subject,
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer_index": int(row["answer"]),
                "answer": row["choices"][int(row["answer"])],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subject_tag = "selected"
    out = args.out_dir / f"{args.split}_{subject_tag}.jsonl"
    write_jsonl(out, rows)
    write_json(
        args.out_dir / f"{args.split}_{subject_tag}_manifest.json",
        {
            "dataset": "cais/mmlu",
            "config": "all",
            "split": args.split,
            "subjects": subjects,
            "limit_per_subject": args.limit_per_subject,
            "examples": len(rows),
            "counts": dict(counts),
            "format": ["id", "subject", "question", "choices", "answer_index", "answer"],
        },
    )
    print(f"Wrote {len(rows)} MMLU examples to {out}")


if __name__ == "__main__":
    main()
