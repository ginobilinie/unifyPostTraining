from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class TaskScore:
    model: str
    task: str
    split: str
    score: float
    role: str = "retention"

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "TaskScore":
        return cls(
            model=str(row["model"]),
            task=str(row["task"]),
            split=str(row.get("split", "test")),
            score=float(row["score"]),
            role=str(row.get("role", "retention")),
        )


@dataclass(frozen=True)
class ForgettingItem:
    model: str
    task: str
    base_score: float
    post_score: float
    forgetting: float
    retention_ratio: float | None


@dataclass(frozen=True)
class ForgettingReport:
    base_model: str
    items: list[ForgettingItem]
    mean_forgetting: float
    mean_retention_ratio: float | None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["items"] = [asdict(item) for item in self.items]
        return payload


def compute_forgetting_report(
    scores: Sequence[TaskScore],
    *,
    base_model: str = "base",
    post_model: str | None = None,
    role: str = "retention",
) -> ForgettingReport:
    base_by_task = {
        score.task: score
        for score in scores
        if score.model == base_model and score.role == role
    }
    post_scores = [
        score
        for score in scores
        if score.model != base_model
        and score.role == role
        and (post_model is None or score.model == post_model)
    ]
    items: list[ForgettingItem] = []
    for post in post_scores:
        base = base_by_task.get(post.task)
        if base is None:
            continue
        ratio = None if base.score == 0 else post.score / base.score
        items.append(
            ForgettingItem(
                model=post.model,
                task=post.task,
                base_score=base.score,
                post_score=post.score,
                forgetting=base.score - post.score,
                retention_ratio=ratio,
            )
        )
    if not items:
        raise ValueError("No matching base/post retention scores found")
    ratios = [item.retention_ratio for item in items if item.retention_ratio is not None]
    return ForgettingReport(
        base_model=base_model,
        items=items,
        mean_forgetting=statistics.mean(item.forgetting for item in items),
        mean_retention_ratio=statistics.mean(ratios) if ratios else None,
    )


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return None
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = (x_var * y_var) ** 0.5
    if denominator == 0:
        return None
    return numerator / denominator
