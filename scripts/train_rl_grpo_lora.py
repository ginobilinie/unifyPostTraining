#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.gsm8k import (
    extract_gold_answer,
    extract_predicted_answer,
    format_gsm8k_prompt,
    read_jsonl,
    write_json,
)
from unify_post_training.hf_runtime import load_causal_lm_base, load_tokenizer, select_device, select_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description="Small on-policy GRPO-style LoRA trainer for GSM8K exact-answer reward.")
    parser.add_argument("--model-path", default="/data/dong/explore/hf_models/Qwen3-0.6B-Base")
    parser.add_argument("--train-file", type=Path, default=Path("data/gsm8k/main_train.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-format", choices=["chat", "plain"], default="plain")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--prompts-per-update", type=int, default=4)
    parser.add_argument("--generations-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-state-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    started = time.time()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = select_device()
    dtype = select_dtype(device)
    tokenizer = load_tokenizer(args.model_path)
    tokenizer.padding_side = "left"
    rows = read_jsonl(args.train_file, limit=args.limit)
    examples = [
        {
            "prompt": format_gsm8k_prompt(tokenizer, row["question"], prompt_format=args.prompt_format),
            "gold": extract_gold_answer(row["answer"]),
        }
        for row in rows
    ]
    if not examples:
        raise SystemExit(f"No examples loaded from {args.train_file}")

    policy_base = load_causal_lm_base(args.model_path, dtype=dtype)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    policy = get_peft_model(policy_base, peft_config).to(device)
    ref = None
    if args.kl_beta > 0:
        ref = load_causal_lm_base(args.model_path, dtype=dtype).to(device)
        ref.eval()
        for param in ref.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW((param for param in policy.parameters() if param.requires_grad), lr=args.lr)
    losses: list[float] = []
    rewards: list[float] = []
    token_kls: list[float] = []
    skipped_no_signal = 0

    for update in range(1, args.updates + 1):
        batch = [examples[rng.randrange(len(examples))] for _ in range(args.prompts_per_update)]
        rollouts = collect_rollouts(policy, tokenizer, device, batch, args)
        if not rollouts:
            continue
        assign_group_advantages(rollouts)
        if not any(abs(float(rollout["advantage"])) > 1e-8 for rollout in rollouts):
            skipped_no_signal += 1
            reward_values = [rollout["reward"] for rollout in rollouts]
            rewards.extend(float(value) for value in reward_values)
            if update == 1 or update % args.log_every == 0:
                print(
                    f"rl_update={update}/{args.updates} "
                    f"reward={sum(reward_values) / len(reward_values):.4f} skipped=no_reward_variance"
                )
            continue
        policy.train()
        input_ids, attention_mask, labels, advantages = rl_batch(rollouts, tokenizer.pad_token_id, device, args.max_state_length)
        policy_logprob, valid_mask = sequence_logprob(policy, input_ids, attention_mask, labels)
        if ref is not None:
            with torch.no_grad():
                ref_logprob, _ = sequence_logprob(ref, input_ids, attention_mask, labels)
            approx_kl = ((policy_logprob - ref_logprob) * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
            kl_loss = args.kl_beta * approx_kl.mean()
            kl_value = float(approx_kl.mean().detach().cpu())
        else:
            kl_loss = policy_logprob.new_tensor(0.0)
            kl_value = 0.0
        seq_logprob = (policy_logprob * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
        loss = -(advantages * seq_logprob).mean() + kl_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((param for param in policy.parameters() if param.requires_grad), 1.0)
        optimizer.step()

        reward_values = [rollout["reward"] for rollout in rollouts]
        losses.append(float(loss.detach().cpu()))
        rewards.extend(float(value) for value in reward_values)
        token_kls.append(kl_value)
        if update == 1 or update % args.log_every == 0:
            print(
                f"rl_update={update}/{args.updates} "
                f"reward={sum(reward_values) / len(reward_values):.4f} "
                f"loss={losses[-1]:.5f} kl={token_kls[-1]:.5f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {
        "trainer": "simple_on_policy_grpo_lora",
        "base_model": args.model_path,
        "train_file": str(args.train_file),
        "prompt_format": args.prompt_format,
        "train_examples": len(examples),
        "updates": args.updates,
        "prompts_per_update": args.prompts_per_update,
        "generations_per_prompt": args.generations_per_prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "lr": args.lr,
        "kl_beta": args.kl_beta,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "final_kl": token_kls[-1] if token_kls else None,
        "mean_kl": sum(token_kls) / len(token_kls) if token_kls else None,
        "skipped_no_signal": skipped_no_signal,
        "seconds": round(time.time() - started, 2),
        "dtype": str(dtype),
    }
    write_json(args.output_dir / "training_metrics.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def collect_rollouts(policy, tokenizer, device: torch.device, examples: list[dict], args) -> list[dict]:
    policy.eval()
    repeated_prompts: list[str] = []
    repeated_gold: list[str] = []
    for example in examples:
        for _ in range(args.generations_per_prompt):
            repeated_prompts.append(example["prompt"])
            repeated_gold.append(example["gold"])
    encoded = tokenizer(repeated_prompts, padding=True, return_tensors="pt").to(device)
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
        outputs = policy.generate(**encoded, **generation_kwargs)
    prompt_width = encoded["input_ids"].shape[1]
    rollouts: list[dict] = []
    for idx in range(outputs.shape[0]):
        prompt_ids = encoded["input_ids"][idx][encoded["attention_mask"][idx].bool()].tolist()
        completion_ids = trim_completion(outputs[idx, prompt_width:].tolist(), tokenizer.pad_token_id, tokenizer.eos_token_id)
        if not completion_ids:
            continue
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        prediction = extract_predicted_answer(completion)
        reward = 1.0 if prediction == repeated_gold[idx] else 0.0
        rollouts.append(
            {
                "group": idx // args.generations_per_prompt,
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "reward": reward,
            }
        )
    return rollouts


def assign_group_advantages(rollouts: list[dict]) -> None:
    rewards_by_group: dict[int, list[float]] = {}
    for rollout in rollouts:
        rewards_by_group.setdefault(rollout["group"], []).append(float(rollout["reward"]))
    means = {group: sum(values) / len(values) for group, values in rewards_by_group.items()}
    for rollout in rollouts:
        rollout["advantage"] = float(rollout["reward"]) - means[rollout["group"]]


def rl_batch(rollouts: list[dict], pad_token_id: int, device: torch.device, max_state_length: int):
    sequences: list[list[int]] = []
    label_sequences: list[list[int]] = []
    advantages: list[float] = []
    for rollout in rollouts:
        completion = rollout["completion_ids"]
        max_prompt_length = max(1, max_state_length - len(completion))
        prompt = rollout["prompt_ids"][-max_prompt_length:]
        sequence = prompt + completion
        labels = [-100] * len(prompt) + completion
        sequences.append(sequence)
        label_sequences.append(labels)
        advantages.append(float(rollout["advantage"]))
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    labels = torch.full((len(sequences), width), -100, dtype=torch.long, device=device)
    for idx, (sequence, label_sequence) in enumerate(zip(sequences, label_sequences)):
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        label_values = torch.tensor(label_sequence, dtype=torch.long, device=device)
        input_ids[idx, -len(sequence) :] = values
        attention_mask[idx, -len(sequence) :] = 1
        labels[idx, -len(sequence) :] = label_values
    return input_ids, attention_mask, labels, torch.tensor(advantages, dtype=torch.float32, device=device)


def sequence_logprob(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels.ne(-100).float()
    safe_labels = shifted_labels.masked_fill(shifted_labels.eq(-100), 0)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logprob = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return token_logprob, valid_mask


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
