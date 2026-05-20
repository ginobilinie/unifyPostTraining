#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.gsm8k import (
    extract_gold_answer,
    extract_predicted_answer,
    format_gsm8k_prompt,
    read_jsonl,
    write_json,
    write_jsonl,
)
from unify_post_training.hf_runtime import load_causal_lm, load_tokenizer, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local causal LM on GSM8K exact final-answer accuracy.")
    parser.add_argument("--model-path", default="/data/dong/explore/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=Path("data/gsm8k/main_test.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/gsm8k_eval/base"))
    parser.add_argument("--model-name", default="base")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-format", choices=["chat", "plain"], default="chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    device = select_device()
    tokenizer = load_tokenizer(args.model_path)
    model = load_causal_lm(args.model_path, adapter_dir=args.adapter_dir, device=device)
    rows = read_jsonl(args.test_file, limit=args.limit)
    started = time.time()
    predictions: list[dict] = []
    correct = 0
    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start : batch_start + args.batch_size]
        prompts = [
            format_gsm8k_prompt(tokenizer, row["question"], prompt_format=args.prompt_format)
            for row in batch
        ]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": args.temperature})
        else:
            generation_kwargs.update({"do_sample": False})
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        prompt_width = inputs["input_ids"].shape[1]
        for offset, row in enumerate(batch):
            idx = batch_start + offset + 1
            completion_ids = output_ids[offset, prompt_width:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            gold = extract_gold_answer(row["answer"])
            pred = extract_predicted_answer(completion)
            is_correct = pred == gold
            correct += int(is_correct)
            predictions.append(
                {
                    "id": row.get("id", f"gsm8k-test-{idx:05d}"),
                    "question": row["question"],
                    "gold": gold,
                    "prediction": pred,
                    "correct": is_correct,
                    "completion": completion,
                }
            )
        evaluated = min(batch_start + len(batch), len(rows))
        if evaluated == len(batch) or evaluated % 25 == 0 or evaluated == len(rows):
            print(f"evaluated={evaluated}/{len(rows)} accuracy={correct / evaluated:.4f}")

    score = correct / max(len(rows), 1)
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
            "accuracy": score,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "prompt_format": args.prompt_format,
            "seconds": round(time.time() - started, 2),
        },
    )
    write_jsonl(
        args.out_dir / "scores.jsonl",
        [
            {
                "model": args.model_name,
                "task": "gsm8k",
                "split": "test",
                "score": score,
                "role": "target",
            }
        ],
    )
    print(f"accuracy={score:.4f} wrote {args.out_dir}")


if __name__ == "__main__":
    main()
