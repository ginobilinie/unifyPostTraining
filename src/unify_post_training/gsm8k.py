from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


ANSWER_RE = re.compile(r"####\s*([^\n]+)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def extract_gold_answer(answer: str) -> str:
    match = ANSWER_RE.search(answer)
    if match:
        return normalize_answer(match.group(1))
    return normalize_answer(answer)


def extract_predicted_answer(text: str) -> str:
    match = ANSWER_RE.search(text)
    if match:
        return normalize_answer(match.group(1))
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return ""
    return normalize_answer(numbers[-1])


def normalize_answer(text: str) -> str:
    text = text.strip().replace(",", "")
    text = re.sub(r"\.$", "", text)
    if "/" in text:
        return text
    try:
        value = float(text)
    except ValueError:
        return text.lower()
    if value.is_integer():
        return str(int(value))
    return str(value)


def sft_messages(question: str, answer: str | None = None) -> list[dict[str, str]]:
    messages = [
        {
            "role": "user",
            "content": "Solve the following math problem step by step.\n\n" + question.strip(),
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages


def format_gsm8k_prompt(tokenizer: Any, question: str, *, prompt_format: str) -> str:
    if prompt_format == "chat":
        return tokenizer.apply_chat_template(
            sft_messages(question),
            tokenize=False,
            add_generation_prompt=True,
        )
    if prompt_format == "plain":
        return f"Problem:\n{question.strip()}\n\nSolution:\n"
    raise ValueError(f"Unsupported prompt_format: {prompt_format}")


def format_gsm8k_training_text(tokenizer: Any, question: str, answer: str, *, prompt_format: str) -> tuple[str, str]:
    if prompt_format == "chat":
        messages = sft_messages(question, answer)
        prompt_messages = messages[:-1]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        return prompt_text, full_text
    if prompt_format == "plain":
        prompt_text = format_gsm8k_prompt(tokenizer, question, prompt_format="plain")
        full_text = prompt_text + answer.strip()
        return prompt_text, full_text
    raise ValueError(f"Unsupported prompt_format: {prompt_format}")
