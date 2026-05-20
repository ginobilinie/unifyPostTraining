#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.gsm8k import read_jsonl, write_json, write_jsonl
from unify_post_training.hf_runtime import load_causal_lm, load_tokenizer, select_device


LETTERS = ["A", "B", "C", "D", "E", "F"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MMLU by multiple-choice conditional log likelihood.")
    parser.add_argument("--model-path", default="/data/dong/explore/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=Path("data/mmlu/test_selected.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/mmlu_eval/base"))
    parser.add_argument("--model-name", default="base")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    device = select_device()
    tokenizer = load_tokenizer(args.model_path)
    model = load_causal_lm(args.model_path, adapter_dir=args.adapter_dir, device=device)
    rows = read_jsonl(args.test_file, limit=args.limit)
    started = time.time()
    predictions: list[dict] = []
    correct = 0
    subject_correct: dict[str, int] = defaultdict(int)
    subject_total: dict[str, int] = defaultdict(int)
    for idx, row in enumerate(rows, start=1):
        scores = [choice_logprob(model, tokenizer, device, row, choice_idx) for choice_idx in range(len(row["choices"]))]
        pred_idx = max(range(len(scores)), key=lambda item: scores[item])
        answer_idx = int(row["answer_index"])
        is_correct = pred_idx == answer_idx
        correct += int(is_correct)
        subject = row.get("subject", "unknown")
        subject_correct[subject] += int(is_correct)
        subject_total[subject] += 1
        predictions.append(
            {
                "id": row["id"],
                "subject": subject,
                "question": row["question"],
                "choices": row["choices"],
                "answer_index": answer_idx,
                "prediction_index": pred_idx,
                "correct": is_correct,
                "choice_logprobs": scores,
            }
        )
        if idx == 1 or idx % 25 == 0:
            print(f"mmlu_evaluated={idx}/{len(rows)} accuracy={correct / idx:.4f}")

    accuracy = correct / max(len(rows), 1)
    by_subject = {
        subject: {
            "correct": subject_correct[subject],
            "total": subject_total[subject],
            "accuracy": subject_correct[subject] / subject_total[subject],
        }
        for subject in sorted(subject_total)
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "predictions.jsonl", predictions)
    write_json(
        args.out_dir / "metrics.json",
        {
            "model": args.model_name,
            "model_path": args.model_path,
            "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
            "test_file": str(args.test_file),
            "examples": len(rows),
            "correct": correct,
            "accuracy": accuracy,
            "by_subject": by_subject,
            "seconds": round(time.time() - started, 2),
            "scoring": "choice_conditional_log_likelihood",
        },
    )
    write_jsonl(
        args.out_dir / "scores.jsonl",
        [
            {
                "model": args.model_name,
                "task": "mmlu_selected",
                "split": "test",
                "score": accuracy,
                "role": "retention",
            }
        ],
    )
    print(f"mmlu_accuracy={accuracy:.4f} wrote {args.out_dir}")


def choice_logprob(model, tokenizer, device: torch.device, row: dict, choice_idx: int) -> float:
    prompt = format_prompt(row)
    target = " " + LETTERS[choice_idx]
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    shifted_targets = input_ids[:, 1:]
    start = max(len(prompt_ids) - 1, 0)
    token_logprobs = log_probs[0, start:, :].gather(1, shifted_targets[0, start:].unsqueeze(1)).squeeze(1)
    if token_logprobs.numel() == 0:
        return -math.inf
    return float(token_logprobs.mean().detach().cpu())


def format_prompt(row: dict) -> str:
    choices = "\n".join(f"{LETTERS[idx]}. {choice}" for idx, choice in enumerate(row["choices"]))
    subject = str(row.get("subject", "general knowledge")).replace("_", " ")
    return (
        f"The following is a multiple-choice question about {subject}.\n\n"
        f"Question: {row['question'].strip()}\n"
        f"{choices}\n"
        "Answer:"
    )


if __name__ == "__main__":
    main()
