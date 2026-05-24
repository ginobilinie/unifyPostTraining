#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.gsm8k import format_gsm8k_prompt, read_jsonl, write_json
from unify_post_training.hf_runtime import load_causal_lm, load_causal_lm_base, load_tokenizer, select_device, select_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description="On-policy distillation: student states, teacher supervision.")
    parser.add_argument("--model-path", default="/data/dong/explore/hf_models/Qwen3-0.6B-Base")
    parser.add_argument("--teacher-adapter-dir", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, default=Path("data/gsm8k/main_train.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-format", choices=["chat", "plain"], default="plain")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--prompts-per-update", type=int, default=4)
    parser.add_argument("--state-batch-size", type=int, default=8)
    parser.add_argument("--max-states-per-update", type=int, default=32)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=96)
    parser.add_argument("--state-stride", type=int, default=8)
    parser.add_argument("--max-state-length", type=int, default=1024)
    parser.add_argument("--state-source", choices=["student", "teacher"], default="student")
    parser.add_argument("--objective", choices=["next_token_kl", "teacher_continuation_ce"], default="teacher_continuation_ce")
    parser.add_argument("--teacher-continuation-max-new-tokens", type=int, default=64)
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    started = time.time()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = select_device()
    dtype = select_dtype(device)
    tokenizer = load_tokenizer(args.model_path)
    rows = read_jsonl(args.prompts_file, limit=args.limit)
    prompts = [format_gsm8k_prompt(tokenizer, row["question"], prompt_format=args.prompt_format) for row in rows]
    if not prompts:
        raise SystemExit(f"No prompts loaded from {args.prompts_file}")

    teacher = load_causal_lm(args.model_path, adapter_dir=args.teacher_adapter_dir, device=device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student_base = load_causal_lm_base(args.model_path, dtype=dtype)
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
    student = get_peft_model(student_base, peft_config).to(device)
    optimizer = torch.optim.AdamW((param for param in student.parameters() if param.requires_grad), lr=args.lr)
    losses: list[float] = []
    states_seen = 0

    for update in range(1, args.updates + 1):
        batch_prompts = [prompts[rng.randrange(len(prompts))] for _ in range(args.prompts_per_update)]
        rollout_policy = student if args.state_source == "student" else teacher
        state_ids = collect_on_policy_states(rollout_policy, tokenizer, device, batch_prompts, args)
        if len(state_ids) > args.max_states_per_update:
            state_ids = [state_ids[index] for index in sorted(rng.sample(range(len(state_ids)), args.max_states_per_update))]
        if not state_ids:
            continue
        student.train()
        update_losses: list[float] = []
        if args.objective == "next_token_kl":
            training_batches = [(chunk, None) for chunk in chunked(state_ids, args.state_batch_size)]
        else:
            training_batches = teacher_continuation_batches(teacher, tokenizer, device, state_ids, args)
        for prefixes, continuations in training_batches:
            if args.objective == "next_token_kl":
                input_ids, attention_mask = left_pad(prefixes, tokenizer.pad_token_id, device)
                with torch.no_grad():
                    teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
                student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
                loss = distillation_kl(student_logits, teacher_logits, args.kl_temperature)
                examples = len(prefixes)
            else:
                assert continuations is not None
                input_ids, attention_mask, labels = continuation_ce_batch(
                    prefixes,
                    continuations,
                    tokenizer.pad_token_id,
                    device,
                    args.max_state_length,
                )
                loss = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                examples = sum(len(continuation) for continuation in continuations)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((param for param in student.parameters() if param.requires_grad), 1.0)
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            update_losses.append(value)
            states_seen += examples
        if update == 1 or update % args.log_every == 0:
            avg = sum(update_losses) / max(len(update_losses), 1)
            print(f"opd_update={update}/{args.updates} states={len(state_ids)} loss={avg:.5f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {
        "trainer": f"opd_{args.objective}",
        "base_model": args.model_path,
        "teacher_adapter_dir": str(args.teacher_adapter_dir),
        "prompts_file": str(args.prompts_file),
        "prompt_format": args.prompt_format,
        "prompt_examples": len(prompts),
        "updates": args.updates,
        "prompts_per_update": args.prompts_per_update,
        "state_batch_size": args.state_batch_size,
        "max_states_per_update": args.max_states_per_update,
        "rollout_max_new_tokens": args.rollout_max_new_tokens,
        "state_stride": args.state_stride,
        "max_state_length": args.max_state_length,
        "state_source": args.state_source,
        "objective": args.objective,
        "teacher_continuation_max_new_tokens": args.teacher_continuation_max_new_tokens,
        "teacher_temperature": args.teacher_temperature,
        "teacher_top_p": args.teacher_top_p,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "kl_temperature": args.kl_temperature,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "states_seen": states_seen,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "seconds": round(time.time() - started, 2),
        "dtype": str(dtype),
    }
    write_json(args.output_dir / "training_metrics.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def collect_on_policy_states(student, tokenizer, device: torch.device, prompts: list[str], args) -> list[list[int]]:
    tokenizer.padding_side = "left"
    student.eval()
    encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": args.rollout_max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        generation_kwargs.update({"do_sample": False})
    with torch.no_grad():
        output_ids = student.generate(**encoded, **generation_kwargs)
    prompt_width = encoded["input_ids"].shape[1]
    states: list[list[int]] = []
    for row_idx in range(output_ids.shape[0]):
        prompt_ids = encoded["input_ids"][row_idx][encoded["attention_mask"][row_idx].bool()].tolist()
        completion_ids = trim_generated_prefix(output_ids[row_idx, prompt_width:].tolist(), tokenizer.pad_token_id, tokenizer.eos_token_id)
        steps = state_steps(len(completion_ids), args.state_stride)
        for step in steps:
            state = prompt_ids + completion_ids[:step]
            if len(state) > args.max_state_length:
                state = state[-args.max_state_length :]
            if state:
                states.append(state)
    return states


def state_steps(length: int, stride: int) -> list[int]:
    if length <= 0:
        return [0]
    steps = list(range(min(stride, length), length + 1, stride))
    if steps[-1] != length:
        steps.append(length)
    return steps


def chunked(values: list[list[int]], size: int) -> list[list[list[int]]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def teacher_continuation_batches(teacher, tokenizer, device: torch.device, state_ids: list[list[int]], args):
    tokenizer.padding_side = "left"
    batches = []
    for prefixes in chunked(state_ids, args.state_batch_size):
        input_ids, attention_mask = left_pad(prefixes, tokenizer.pad_token_id, device)
        generation_kwargs = {
            "max_new_tokens": args.teacher_continuation_max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.teacher_temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": args.teacher_temperature, "top_p": args.teacher_top_p})
        else:
            generation_kwargs.update({"do_sample": False})
        with torch.no_grad():
            outputs = teacher.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
        prompt_width = input_ids.shape[1]
        kept_prefixes: list[list[int]] = []
        continuations: list[list[int]] = []
        for row_idx, prefix in enumerate(prefixes):
            continuation = trim_continuation(outputs[row_idx, prompt_width:].tolist(), tokenizer.pad_token_id, tokenizer.eos_token_id)
            if continuation:
                kept_prefixes.append(prefix)
                continuations.append(continuation)
        if kept_prefixes:
            batches.append((kept_prefixes, continuations))
    return batches


def trim_continuation(tokens: list[int], pad_token_id: int, eos_token_id: int | None) -> list[int]:
    result: list[int] = []
    for token in tokens:
        if token == pad_token_id:
            break
        result.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
    return result


def trim_generated_prefix(tokens: list[int], pad_token_id: int, eos_token_id: int | None) -> list[int]:
    result: list[int] = []
    for token in tokens:
        if token == pad_token_id:
            break
        if eos_token_id is not None and token == eos_token_id:
            break
        result.append(token)
    return result


def continuation_ce_batch(
    prefixes: list[list[int]],
    continuations: list[list[int]],
    pad_token_id: int,
    device: torch.device,
    max_state_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences: list[list[int]] = []
    label_sequences: list[list[int]] = []
    for prefix, continuation in zip(prefixes, continuations):
        max_prefix_length = max(1, max_state_length - len(continuation))
        clipped_prefix = prefix[-max_prefix_length:]
        sequence = clipped_prefix + continuation
        labels = [-100] * len(clipped_prefix) + continuation
        sequences.append(sequence)
        label_sequences.append(labels)

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
    return input_ids, attention_mask, labels


def left_pad(sequences: list[list[int]], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for idx, sequence in enumerate(sequences):
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        input_ids[idx, -len(sequence) :] = values
        attention_mask[idx, -len(sequence) :] = 1
    return input_ids, attention_mask


def distillation_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = max(temperature, 1e-6)
    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)


if __name__ == "__main__":
    main()
