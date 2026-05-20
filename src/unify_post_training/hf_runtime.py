from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unify_post_training.lora import load_lora_adapter


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(model_path: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(
    model_path: str | Path,
    *,
    adapter_dir: str | Path | None = None,
    device: torch.device | None = None,
):
    device = device or select_device()
    dtype = select_dtype(device)
    model = load_causal_lm_base(model_path, dtype=dtype)
    if adapter_dir is not None:
        adapter_path = Path(adapter_dir)
        if (adapter_path / "adapter.pt").exists():
            load_lora_adapter(model, adapter_path, map_location="cpu")
        else:
            model = load_peft_adapter(model, adapter_path)
    model.to(device)
    model.eval()
    return model


def load_causal_lm_base(model_path: str | Path, *, dtype: torch.dtype):
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    try:
        return AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype, **kwargs)


def load_peft_adapter(model, adapter_dir: Path):
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit(
            f"{adapter_dir} looks like a PEFT adapter, but peft is not installed. "
            "Run this script from the qwen_sft or trl conda environment."
        ) from exc
    return PeftModel.from_pretrained(model, str(adapter_dir))
