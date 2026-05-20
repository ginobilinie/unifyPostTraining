#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.gsm8k import format_gsm8k_prompt, read_jsonl, write_jsonl
from unify_post_training.hf_runtime import load_causal_lm, load_tokenizer, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect autoregressive prefix states from model rollouts.")
    parser.add_argument("--model-path", default="/data/dong/explore/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--prompts-file", type=Path, default=Path("data/gsm8k/main_test.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", default="base")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--state-stride", type=int, default=8)
    parser.add_argument("--prompt-format", choices=["chat", "plain"], default="chat")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device()
    tokenizer = load_tokenizer(args.model_path)
    model = load_causal_lm(args.model_path, adapter_dir=args.adapter_dir, device=device)
    rows = read_jsonl(args.prompts_file, limit=args.limit)
    states: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        prompt_id = row.get("id", f"prompt-{idx:05d}")
        prompt = format_gsm8k_prompt(tokenizer, row["question"], prompt_format=args.prompt_format)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
            output_ids = model.generate(**inputs, **generation_kwargs)
        completion_ids = output_ids[0, inputs["input_ids"].shape[1] :].tolist()
        trajectory_id = stable_id(args.model_name, prompt_id, completion_ids)
        steps = state_steps(len(completion_ids), args.state_stride)
        for step in steps:
            prefix_ids = completion_ids[:step]
            prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True)
            states.append(
                {
                    "model": args.model_name,
                    "prompt_id": prompt_id,
                    "trajectory_id": trajectory_id,
                    "step": step,
                    "prompt": row["question"],
                    "prompt_format": args.prompt_format,
                    "prefix": prefix,
                    "embedding": lexical_embedding(row["question"] + "\n" + prefix),
                }
            )
        if idx == 1 or idx % 25 == 0:
            print(f"collected_prompts={idx}/{len(rows)} states={len(states)}")
    write_jsonl(args.out, states)
    print(f"wrote {len(states)} states to {args.out}")


def state_steps(length: int, stride: int) -> list[int]:
    if length <= 0:
        return [0]
    steps = list(range(min(stride, length), length + 1, stride))
    if steps[-1] != length:
        steps.append(length)
    return steps


def stable_id(model_name: str, prompt_id: str, completion_ids: list[int]) -> str:
    digest = hashlib.sha1((" ".join(map(str, completion_ids))).encode("utf-8")).hexdigest()[:12]
    return f"{model_name}:{prompt_id}:{digest}"


def lexical_embedding(text: str, dim: int = 128) -> list[float]:
    values = [0.0] * dim
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[bucket] += sign
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0:
        return values
    return [value / norm for value in values]


if __name__ == "__main__":
    main()
