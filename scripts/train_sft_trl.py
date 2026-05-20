#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.gsm8k import format_gsm8k_training_text, read_jsonl, write_json
from unify_post_training.hf_runtime import load_causal_lm_base, select_device, select_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description="TRL/PEFT LoRA SFT for GSM8K chat JSONL.")
    parser.add_argument("--model-path", default="/data/dong/explore/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=Path("data/gsm8k/main_train_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen2_5_0_5b_gsm8k_sft_peft"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--assistant-only-loss", action="store_true")
    parser.add_argument("--prompt-format", choices=["chat", "plain"], default="chat")
    args = parser.parse_args()
    print(
        "SFT config: "
        f"batch_size={args.batch_size} "
        f"grad_accum={args.grad_accum} "
        f"effective_batch={args.batch_size * args.grad_accum} "
        f"limit={args.limit} epochs={args.epochs} max_length={args.max_length} "
        f"lr={args.lr} lora_rank={args.lora_rank} "
        f"lora_alpha={args.lora_alpha} lora_dropout={args.lora_dropout} "
        f"prompt_format={args.prompt_format}"
    )

    device = select_device()
    dtype = select_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(args.train_file, limit=args.limit)
    dataset = Dataset.from_list(to_training_rows(rows, tokenizer, args.prompt_format))
    model = load_causal_lm_base(args.model_path, dtype=dtype)

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
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        bf16=(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        fp16=(device.type == "cuda" and not torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,
        report_to=[],
        assistant_only_loss=args.assistant_only_loss,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {
        "trainer": "trl.SFTTrainer",
        "base_model": args.model_path,
        "train_file": str(args.train_file),
        "train_examples": len(rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "assistant_only_loss": args.assistant_only_loss,
        "prompt_format": args.prompt_format,
        "dtype": str(dtype),
    }
    write_json(args.output_dir / "training_metrics.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def to_training_rows(rows: list[dict], tokenizer, prompt_format: str) -> list[dict[str, str]]:
    if prompt_format == "chat":
        return [{"messages": row["messages"]} for row in rows]
    return [
        {"text": format_gsm8k_training_text(tokenizer, row["question"], row["answer"], prompt_format=prompt_format)[1]}
        for row in rows
    ]


if __name__ == "__main__":
    main()
