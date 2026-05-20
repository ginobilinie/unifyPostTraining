#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.states import StateRecord, write_jsonl


PROMPTS = [
    "Solve the arithmetic word problem step by step.",
    "Write a Python function for the requested transformation.",
    "Answer the factual question with a concise explanation.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic state-drift fixture.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = make_states("base", "careful reasoning answer", 0.0, rng)
    post = make_states("post", "specialized math proof answer", 0.7, rng)
    scores = [
        {"model": "base", "task": "mmlu", "score": 0.64, "role": "retention"},
        {"model": "base", "task": "gsm8k", "score": 0.52, "role": "retention"},
        {"model": "post", "task": "mmlu", "score": 0.58, "role": "retention"},
        {"model": "post", "task": "gsm8k", "score": 0.49, "role": "retention"},
        {"model": "post", "task": "math", "score": 0.35, "role": "target"},
    ]
    write_jsonl(args.out_dir / "base_states.jsonl", (state.to_json() for state in base))
    write_jsonl(args.out_dir / "post_states.jsonl", (state.to_json() for state in post))
    write_jsonl(args.out_dir / "scores.jsonl", scores)
    print(f"Wrote toy fixtures under {args.out_dir}")


def make_states(model: str, phrase: str, shift: float, rng: random.Random) -> list[StateRecord]:
    states: list[StateRecord] = []
    words = phrase.split()
    for prompt_idx, prompt in enumerate(PROMPTS):
        prefix = ""
        for step in range(1, 9):
            prefix = f"{prefix} {words[step % len(words)]}".strip()
            angle = step / 3.0
            embedding = [
                math.sin(angle) + shift + rng.uniform(-0.03, 0.03),
                math.cos(angle) + (shift * 0.5) + rng.uniform(-0.03, 0.03),
                (prompt_idx / 3.0) + rng.uniform(-0.03, 0.03),
            ]
            states.append(
                StateRecord(
                    model=model,
                    prompt_id=f"toy-{prompt_idx}",
                    trajectory_id=f"{model}:toy-{prompt_idx}:0",
                    step=step,
                    prompt=prompt,
                    prefix=f"{prompt} {prefix}",
                    embedding=embedding,
                )
            )
    return states


if __name__ == "__main__":
    main()
