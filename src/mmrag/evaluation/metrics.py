"""Small, dependency-light metrics with explicit edge-case semantics."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class RetrievalMetrics:
    recall: dict[int, float]
    precision: dict[int, float]
    mrr: float


def retrieval_metrics(
    ranked_ids: list[str], relevant_ids: set[str], ks: tuple[int, ...] = (1, 3, 5, 10)
) -> RetrievalMetrics:
    if any(k <= 0 for k in ks):
        raise ValueError("K values must be positive")
    recall: dict[int, float] = {}
    precision: dict[int, float] = {}
    for k in ks:
        retrieved = ranked_ids[:k]
        hits = sum(item in relevant_ids for item in retrieved)
        recall[k] = hits / len(relevant_ids) if relevant_ids else 0.0
        precision[k] = hits / k
    first_rank = next((rank for rank, item in enumerate(ranked_ids, 1) if item in relevant_ids), None)
    return RetrievalMetrics(recall=recall, precision=precision, mrr=1 / first_rank if first_rank else 0.0)


_NUMBER_PATTERN = re.compile(r"[-+]?\(?\s*[₹$€£]?\s*[\d,]+(?:\.\d+)?\s*\)?%?")


def _as_decimal(text: str) -> Decimal | None:
    match = _NUMBER_PATTERN.search(text)
    if not match:
        return None
    raw = match.group(0).replace(",", "").replace(" ", "")
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").lstrip("₹$€£").rstrip("%")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    return -value if negative else value


def numeric_exact_match(prediction: str, reference: str, tolerance: float = 1e-4) -> float:
    pred = _as_decimal(prediction)
    ref = _as_decimal(reference)
    if pred is None or ref is None:

        def normalise(value: str) -> str:
            return " ".join(value.lower().split())

        return float(normalise(prediction) == normalise(reference))
    allowed = Decimal(str(tolerance)) * max(abs(ref), Decimal(1))
    return float(abs(pred - ref) <= allowed)


def bootstrap_mean_ci(
    values: list[float], *, confidence: float = 0.95, samples: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty sample")
    if not 0 < confidence < 1 or samples <= 0:
        raise ValueError("confidence must be between 0 and 1 and samples must be positive")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    alpha = (1 - confidence) / 2
    low_index = max(0, math.floor(alpha * (samples - 1)))
    high_index = min(samples - 1, math.ceil((1 - alpha) * (samples - 1)))
    return mean, boot[low_index], boot[high_index]


def paired_permutation_test(
    baseline: list[float], candidate: list[float], *, permutations: int = 10000, seed: int = 42
) -> float:
    """Two-sided paired randomisation test, exact for at most 16 non-zero pairs."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired samples must be non-empty and have equal length")
    differences = [right - left for left, right in zip(baseline, candidate, strict=True)]
    differences = [value for value in differences if value != 0]
    if not differences:
        return 1.0
    observed = abs(sum(differences) / len(differences))
    if len(differences) <= 16:
        totals = 1 << len(differences)
        extreme = 0
        for mask in range(totals):
            value = sum(
                difference if mask & (1 << index) else -difference for index, difference in enumerate(differences)
            ) / len(differences)
            extreme += abs(value) >= observed - 1e-15
        return extreme / totals
    rng = random.Random(seed)
    extreme = 0
    for _ in range(permutations):
        value = sum(value if rng.random() < 0.5 else -value for value in differences) / len(differences)
        extreme += abs(value) >= observed - 1e-15
    return (extreme + 1) / (permutations + 1)
