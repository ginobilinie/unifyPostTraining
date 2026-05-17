#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.drift import compute_drift_report
from unify_post_training.states import read_states, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure drift between two state JSONL files.")
    parser.add_argument("--left", type=Path, required=True, help="Reference state JSONL, often base model.")
    parser.add_argument("--right", type=Path, required=True, help="Comparison state JSONL.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON report.")
    parser.add_argument("--mmd-gamma", type=float, default=None)
    parser.add_argument("--wasserstein-projections", type=int, default=32)
    args = parser.parse_args()

    report = compute_drift_report(
        read_states(args.left),
        read_states(args.right),
        mmd_gamma=args.mmd_gamma,
        wasserstein_projections=args.wasserstein_projections,
    )
    write_json(args.out, report.to_json())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
