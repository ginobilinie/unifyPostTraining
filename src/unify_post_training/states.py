from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class StateRecord:
    model: str
    prompt_id: str
    trajectory_id: str
    step: int
    prompt: str
    prefix: str
    embedding: list[float] | None = None

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "StateRecord":
        return cls(
            model=str(row.get("model", "")),
            prompt_id=str(row.get("prompt_id", "")),
            trajectory_id=str(row.get("trajectory_id", "")),
            step=int(row.get("step", 0)),
            prompt=str(row.get("prompt", "")),
            prefix=str(row.get("prefix", "")),
            embedding=normalize_embedding(row.get("embedding")),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def normalize_embedding(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("embedding must be a JSON list of numbers")
    return [float(item) for item in value]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_states(path: Path) -> list[StateRecord]:
    return [StateRecord.from_json(row) for row in read_jsonl(path)]
