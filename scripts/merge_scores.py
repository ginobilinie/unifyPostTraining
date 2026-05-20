#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge target and retention metrics into one score JSONL.")
    parser.add_argument("--base-metrics", type=Path, required=True, help="Base GSM8K metrics.")
    parser.add_argument("--post-metrics", type=Path, required=True, help="Post/SFT GSM8K metrics.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--role", default="target")
    parser.add_argument("--post-model", default="sft")
    parser.add_argument("--base-retention-metrics", type=Path, default=None)
    parser.add_argument("--post-retention-metrics", type=Path, default=None)
    parser.add_argument("--retention-task", default="truthfulqa_mc1")
    parser.add_argument(
        "--retention-pair",
        nargs=3,
        action="append",
        metavar=("BASE_METRICS", "POST_METRICS", "TASK"),
        default=[],
        help="Additional retention pair: base_metrics post_metrics task_name.",
    )
    args = parser.parse_args()

    rows = [
        score_row(read_json(args.base_metrics), "base", args.task, args.role),
        score_row(read_json(args.post_metrics), args.post_model, args.task, args.role),
    ]
    if (
        args.base_retention_metrics
        and args.post_retention_metrics
        and args.base_retention_metrics.exists()
        and args.post_retention_metrics.exists()
    ):
        rows.extend(
            [
                score_row(read_json(args.base_retention_metrics), "base", args.retention_task, "retention"),
                score_row(read_json(args.post_retention_metrics), args.post_model, args.retention_task, "retention"),
            ]
        )
    for base_path, post_path, task in args.retention_pair:
        base_metrics = Path(base_path)
        post_metrics = Path(post_path)
        if base_metrics.exists() and post_metrics.exists():
            rows.extend(
                [
                    score_row(read_json(base_metrics), "base", task, "retention"),
                    score_row(read_json(post_metrics), args.post_model, task, "retention"),
                ]
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {args.out}")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def score_row(metrics: dict, model: str, task: str, role: str) -> dict:
    return {
        "model": model,
        "task": task,
        "split": metrics.get("split", "test"),
        "score": metrics["accuracy"],
        "role": role,
    }


if __name__ == "__main__":
    main()
