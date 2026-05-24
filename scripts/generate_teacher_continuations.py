#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.gsm8k import extract_predicted_answer, format_gsm8k_prompt, read_jsonl, write_json
from unify_post_training.hf_runtime import load_causal_lm, load_tokenizer, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher continuations as sequence-level KD/SFT data.")
    parser.add_argument("--model-path", default="/data/dong/explore/hf_models/Qwen3-0.6B-Base")
    parser.add_argument("--teacher-adapter-dir", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, default=Path("data/gsm8k/main_train.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-format", choices=["chat", "plain"], default="plain")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--model-name", default="teacher")
    args = parser.parse_args()

    device = select_device()
    tokenizer = load_tokenizer(args.model_path)
    tokenizer.padding_side = "left"
    teacher = load_causal_lm(args.model_path, adapter_dir=args.teacher_adapter_dir, device=device)
    teacher.eval()

    rows = read_jsonl(args.prompts_file, limit=args.limit)
    out_rows: list[dict] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [format_gsm8k_prompt(tokenizer, row["question"], prompt_format=args.prompt_format) for row in batch]
        encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
        else:
            generation_kwargs.update({"do_sample": False})
        with torch.no_grad():
            outputs = teacher.generate(**encoded, **generation_kwargs)
        prompt_width = encoded["input_ids"].shape[1]
        for row_idx, row in enumerate(batch):
            completion_ids = trim_completion(outputs[row_idx, prompt_width:].tolist(), tokenizer.pad_token_id, tokenizer.eos_token_id)
            answer = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            out_rows.append(
                {
                    "id": row.get("id", f"teacher-{len(out_rows):05d}"),
                    "split": row.get("split", "train"),
                    "question": row["question"],
                    "answer": answer,
                    "final_answer": extract_predicted_answer(answer),
                    "teacher_model": args.model_name,
                    "teacher_adapter_dir": str(args.teacher_adapter_dir),
                    "messages": [
                        {
                            "role": "user",
                            "content": "Solve the following math problem step by step.\n\n" + row["question"].strip(),
                        },
                        {"role": "assistant", "content": answer},
                    ],
                }
            )
        print(f"generated={len(out_rows)}/{len(rows)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(
        args.out.with_suffix(".manifest.json"),
        {
            "model_path": args.model_path,
            "teacher_adapter_dir": str(args.teacher_adapter_dir),
            "prompts_file": str(args.prompts_file),
            "out": str(args.out),
            "prompt_format": args.prompt_format,
            "examples": len(out_rows),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )
    print(f"wrote {len(out_rows)} teacher continuations to {args.out}")


def trim_completion(tokens: list[int], pad_token_id: int, eos_token_id: int | None) -> list[int]:
    result: list[int] = []
    for token in tokens:
        if token == pad_token_id:
            break
        if eos_token_id is not None and token == eos_token_id:
            break
        result.append(token)
    return result


if __name__ == "__main__":
    main()
