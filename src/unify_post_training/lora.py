from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    @classmethod
    def from_json(cls, path: Path) -> "LoraConfig":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload.get("target_modules"), list):
            payload["target_modules"] = tuple(payload["target_modules"])
        return cls(**payload)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: LoraConfig) -> None:
        super().__init__()
        self.base = base
        self.rank = config.rank
        self.alpha = config.alpha
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)
        self.lora_a = nn.Linear(base.in_features, config.rank, bias=False)
        self.lora_b = nn.Linear(config.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def apply_lora(model: nn.Module, config: LoraConfig) -> list[str]:
    replaced: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if should_replace(child_name, full_name, child, config.target_modules):
                setattr(module, child_name, LoRALinear(child, config))
                replaced.append(full_name)
    if not replaced:
        raise ValueError(f"No target Linear modules matched {config.target_modules}")
    return replaced


def should_replace(
    child_name: str,
    full_name: str,
    child: nn.Module,
    target_modules: Iterable[str],
) -> bool:
    if not isinstance(child, nn.Linear):
        return False
    return any(child_name == target or full_name.endswith(f".{target}") for target in target_modules)


def mark_only_lora_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        param.requires_grad = "lora_a" in name or "lora_b" in name
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "lora_a" in name or "lora_b" in name
    }


def save_lora_adapter(model: nn.Module, config: LoraConfig, output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), output_dir / "adapter.pt")
    with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_json(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "adapter_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_lora_adapter(model: nn.Module, adapter_dir: Path, map_location: str | torch.device = "cpu") -> LoraConfig:
    config = LoraConfig.from_json(adapter_dir / "adapter_config.json")
    apply_lora(model, config)
    state = torch.load(adapter_dir / "adapter.pt", map_location=map_location)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_lora = [name for name in unexpected if "lora_" in name]
    missing_lora = [name for name in missing if "lora_" in name]
    if unexpected_lora or missing_lora:
        raise ValueError(
            f"Adapter mismatch. Missing LoRA keys: {missing_lora[:5]}, "
            f"unexpected LoRA keys: {unexpected_lora[:5]}"
        )
    return config
