#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.metrics import TaskScore, compute_forgetting_report
from unify_post_training.states import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze retention and attach a state drift report.")
    parser.add_argument("--scores", type=Path, required=True, help="JSONL score table.")
    parser.add_argument("--drift", type=Path, required=True, help="State drift JSON report.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-model", default="base")
    parser.add_argument("--post-model", default=None)
    args = parser.parse_args()

    scores = [TaskScore.from_json(row) for row in read_jsonl(args.scores)]
    forgetting = compute_forgetting_report(
        scores,
        base_model=args.base_model,
        post_model=args.post_model,
    )
    drift = read_json(args.drift)
    payload: dict[str, Any] = {
        "forgetting": forgetting.to_json(),
        "drift": drift,
        "primary_point": {
            "model": drift.get("right_model"),
            "x_state_drift": drift.get("mmd_rbf") or drift.get("centroid_l2") or drift.get("lexical_jaccard"),
            "y_mean_forgetting": forgetting.mean_forgetting,
            "mean_retention_ratio": forgetting.mean_retention_ratio,
        },
    }
    write_json(args.out, payload)
    print(f"Wrote {args.out}")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
