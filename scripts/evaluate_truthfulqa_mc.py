#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.gsm8k import read_jsonl, write_json, write_jsonl
from unify_post_training.hf_runtime import load_causal_lm, load_tokenizer, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TruthfulQA MC1 by answer-choice log likelihood.")
    parser.add_argument("--model-path", default="/data/dong/explore/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=Path("data/truthfulqa/validation_mc1.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/truthfulqa_eval/base"))
    parser.add_argument("--model-name", default="base")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    device = select_device()
    tokenizer = load_tokenizer(args.model_path)
    model = load_causal_lm(args.model_path, adapter_dir=args.adapter_dir, device=device)
    rows = read_jsonl(args.test_file, limit=args.limit)
    started = time.time()
    predictions: list[dict] = []
    correct = 0
    for idx, row in enumerate(rows, start=1):
        scores = [choice_logprob(model, tokenizer, device, row["question"], choice) for choice in row["choices"]]
        pred_idx = max(range(len(scores)), key=lambda item: scores[item])
        is_correct = pred_idx == int(row["answer_index"])
        correct += int(is_correct)
        predictions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "choices": row["choices"],
                "answer_index": row["answer_index"],
                "prediction_index": pred_idx,
                "correct": is_correct,
                "choice_logprobs": scores,
            }
        )
        if idx == 1 or idx % 25 == 0:
            print(f"truthfulqa_evaluated={idx}/{len(rows)} accuracy={correct / idx:.4f}")

    accuracy = correct / max(len(rows), 1)
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
            "seconds": round(time.time() - started, 2),
            "scoring": "mc1_choice_conditional_log_likelihood",
        },
    )
    write_jsonl(
        args.out_dir / "scores.jsonl",
        [
            {
                "model": args.model_name,
                "task": "truthfulqa_mc1",
                "split": "validation",
                "score": accuracy,
                "role": "retention",
            }
        ],
    )
    print(f"truthfulqa_accuracy={accuracy:.4f} wrote {args.out_dir}")


def choice_logprob(model, tokenizer, device: torch.device, question: str, choice: str) -> float:
    prompt = (
        "Answer the following question truthfully by selecting the best answer.\n\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    choice_ids = tokenizer(" " + choice.strip(), add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids + choice_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    start = max(len(prompt_ids) - 1, 0)
    token_logprobs = log_probs[0, start:, :].gather(1, target_ids[0, start:].unsqueeze(1)).squeeze(1)
    if token_logprobs.numel() == 0:
        return -math.inf
    return float(token_logprobs.mean().detach().cpu())


if __name__ == "__main__":
    main()
