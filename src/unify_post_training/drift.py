from __future__ import annotations

import math
import random
import re
from dataclasses import asdict, dataclass
from typing import Sequence

from unify_post_training.states import StateRecord


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class DriftReport:
    left_model: str
    right_model: str
    left_states: int
    right_states: int
    lexical_jaccard: float
    centroid_l2: float | None
    mmd_rbf: float | None
    sliced_wasserstein: float | None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def compute_drift_report(
    left: Sequence[StateRecord],
    right: Sequence[StateRecord],
    *,
    mmd_gamma: float | None = None,
    wasserstein_projections: int = 32,
    seed: int = 13,
) -> DriftReport:
    if not left or not right:
        raise ValueError("Both state sets must be non-empty")

    left_embeddings, right_embeddings = aligned_embeddings(left, right)
    centroid = None
    mmd = None
    wasserstein = None
    if left_embeddings and right_embeddings:
        centroid = centroid_l2(left_embeddings, right_embeddings)
        mmd = mmd_rbf(left_embeddings, right_embeddings, gamma=mmd_gamma)
        wasserstein = sliced_wasserstein(
            left_embeddings,
            right_embeddings,
            projections=wasserstein_projections,
            seed=seed,
        )

    return DriftReport(
        left_model=first_model(left),
        right_model=first_model(right),
        left_states=len(left),
        right_states=len(right),
        lexical_jaccard=lexical_jaccard_distance(left, right),
        centroid_l2=centroid,
        mmd_rbf=mmd,
        sliced_wasserstein=wasserstein,
    )


def first_model(states: Sequence[StateRecord]) -> str:
    return states[0].model or "unknown"


def lexical_jaccard_distance(left: Sequence[StateRecord], right: Sequence[StateRecord]) -> float:
    left_tokens = token_set(record.prefix for record in left)
    right_tokens = token_set(record.prefix for record in right)
    if not left_tokens and not right_tokens:
        return 0.0
    return 1.0 - (len(left_tokens & right_tokens) / len(left_tokens | right_tokens))


def token_set(texts: Sequence[str] | object) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        tokens.update(token.lower() for token in TOKEN_RE.findall(str(text)))
    return tokens


def aligned_embeddings(
    left: Sequence[StateRecord],
    right: Sequence[StateRecord],
) -> tuple[list[list[float]], list[list[float]]]:
    left_embeddings = [record.embedding for record in left if record.embedding is not None]
    right_embeddings = [record.embedding for record in right if record.embedding is not None]
    if not left_embeddings or not right_embeddings:
        return [], []
    dim = len(left_embeddings[0])
    if dim == 0:
        return [], []
    if any(len(vec) != dim for vec in left_embeddings + right_embeddings):
        raise ValueError("All embeddings must have the same non-zero dimensionality")
    return left_embeddings, right_embeddings


def centroid_l2(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    left_centroid = mean_vector(left)
    right_centroid = mean_vector(right)
    return l2(left_centroid, right_centroid)


def mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    dim = len(vectors[0])
    return [sum(vector[idx] for vector in vectors) / len(vectors) for idx in range(dim)]


def l2(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def mmd_rbf(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    gamma: float | None = None,
) -> float:
    if gamma is None:
        gamma = median_heuristic_gamma([*left, *right])
    xx = average_kernel(left, left, gamma)
    yy = average_kernel(right, right, gamma)
    xy = average_kernel(left, right, gamma)
    return max(0.0, xx + yy - (2.0 * xy))


def median_heuristic_gamma(vectors: Sequence[Sequence[float]]) -> float:
    distances: list[float] = []
    for idx, left in enumerate(vectors):
        for right in vectors[idx + 1 :]:
            distances.append(sum((a - b) ** 2 for a, b in zip(left, right)))
    positive = sorted(distance for distance in distances if distance > 0)
    if not positive:
        return 1.0
    median = positive[len(positive) // 2]
    return 1.0 / max(median, 1e-12)


def average_kernel(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    gamma: float,
) -> float:
    total = 0.0
    count = 0
    for left_vec in left:
        for right_vec in right:
            total += math.exp(-gamma * sum((a - b) ** 2 for a, b in zip(left_vec, right_vec)))
            count += 1
    return total / count


def sliced_wasserstein(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    projections: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    dim = len(left[0])
    total = 0.0
    for _ in range(projections):
        direction = random_unit_vector(dim, rng)
        left_projection = sorted(dot(vec, direction) for vec in left)
        right_projection = sorted(dot(vec, direction) for vec in right)
        total += one_dimensional_wasserstein(left_projection, right_projection)
    return total / projections


def random_unit_vector(dim: int, rng: random.Random) -> list[float]:
    values = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def one_dimensional_wasserstein(left: Sequence[float], right: Sequence[float]) -> float:
    n = max(len(left), len(right))
    if n == 0:
        return 0.0
    total = 0.0
    for idx in range(n):
        left_idx = min(round(idx * (len(left) - 1) / max(n - 1, 1)), len(left) - 1)
        right_idx = min(round(idx * (len(right) - 1) / max(n - 1, 1)), len(right) - 1)
        total += abs(left[left_idx] - right[right_idx])
    return total / n
