#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
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
    parser.add_argument("--max-states", type=int, default=1000, help="Deterministically subsample each side for expensive drift metrics.")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    left = sample_states(read_states(args.left), args.max_states, args.seed)
    right = sample_states(read_states(args.right), args.max_states, args.seed + 1)
    report = compute_drift_report(
        left,
        right,
        mmd_gamma=args.mmd_gamma,
        wasserstein_projections=args.wasserstein_projections,
        seed=args.seed,
    )
    write_json(args.out, report.to_json())
    print(f"Wrote {args.out}")


def sample_states(states: list, max_states: int, seed: int) -> list:
    if max_states <= 0 or len(states) <= max_states:
        return states
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(states)), max_states))
    return [states[index] for index in indices]


if __name__ == "__main__":
    main()
