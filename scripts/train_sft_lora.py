#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.gsm8k import format_gsm8k_training_text, read_jsonl, write_json
from unify_post_training.hf_runtime import load_causal_lm_base, load_tokenizer, select_device, select_dtype
from unify_post_training.lora import LoraConfig, apply_lora, mark_only_lora_trainable, save_lora_adapter


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int, prompt_format: str) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt_text, full_text = format_gsm8k_training_text(
            self.tokenizer,
            row["question"],
            row["answer"],
            prompt_format=self.prompt_format,
        )
        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids[: self.max_length]
        prompt_len = min(len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids), len(full_ids))
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return {"input_ids": input_ids, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT on GSM8K chat JSONL without PEFT/TRL.")
    parser.add_argument("--model-path", default="/data/dong/explore/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=Path("data/gsm8k/main_train_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen2_5_0_5b_gsm8k_sft_lora"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--prompt-format", choices=["chat", "plain"], default="chat")
    args = parser.parse_args()

    device = select_device()
    dtype = select_dtype(device)
    tokenizer = load_tokenizer(args.model_path)
    rows = read_jsonl(args.train_file, limit=args.limit)
    dataset = SFTDataset(rows, tokenizer, args.max_length, args.prompt_format)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))

    model = load_causal_lm_base(args.model_path, dtype=dtype)
    lora_config = LoraConfig(rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    replaced = apply_lora(model, lora_config)
    trainable, total = mark_only_lora_trainable(model)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    total_updates = math.ceil(len(loader) * args.epochs / args.grad_accum)
    started = time.time()
    losses: list[float] = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            losses.append(float(loss.detach().cpu()) * args.grad_accum)
            if batch_idx % args.grad_accum == 0 or batch_idx == len(loader):
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 1 or step % args.log_every == 0:
                    recent = sum(losses[-args.log_every :]) / min(len(losses), args.log_every)
                    print(f"epoch={epoch + 1} update={step}/{total_updates} loss={recent:.4f}")

    metadata = {
        "base_model": args.model_path,
        "train_file": str(args.train_file),
        "train_examples": len(rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "lr": args.lr,
        "prompt_format": args.prompt_format,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "target_modules_replaced": replaced,
        "seconds": round(time.time() - started, 2),
        "final_loss": losses[-1] if losses else None,
    }
    save_lora_adapter(model, lora_config, args.output_dir, metadata)
    write_json(args.output_dir / "training_metrics.json", metadata)
    print(f"Saved LoRA adapter to {args.output_dir}")


def collate(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if __name__ == "__main__":
    main()
