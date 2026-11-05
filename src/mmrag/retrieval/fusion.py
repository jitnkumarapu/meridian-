"""Rank fusion without comparing incompatible raw score spaces."""

from __future__ import annotations

from dataclasses import dataclass, field

from mmrag.indexing import ScoredItem


@dataclass
class FusionItem:
    item_id: str
    score: float = 0.0
    rank_by_source: dict[str, int] = field(default_factory=dict)
    raw_score_by_source: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(rankings: dict[str, list[ScoredItem]], *, rrf_k: int = 60) -> list[FusionItem]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    fused: dict[str, FusionItem] = {}
    for source, ranking in rankings.items():
        for rank, item in enumerate(ranking, start=1):
            result = fused.setdefault(item.item_id, FusionItem(item.item_id))
            result.score += 1.0 / (rrf_k + rank)
            result.rank_by_source[source] = rank
            result.raw_score_by_source[source] = item.score
    return sorted(fused.values(), key=lambda item: (-item.score, item.item_id))


def weighted_rank_fusion(rankings: dict[str, list[ScoredItem]], *, weights: dict[str, float]) -> list[FusionItem]:
    """Fuse independently min-max-normalized scores from every retrieval source."""

    fused: dict[str, FusionItem] = {}
    for source, ranking in rankings.items():
        if not ranking:
            continue
        values = [item.score for item in ranking]
        low, high = min(values), max(values)
        for rank, item in enumerate(ranking, start=1):
            normalized = (item.score - low) / (high - low) if high > low else 1.0
            result = fused.setdefault(item.item_id, FusionItem(item.item_id))
            result.score += weights.get(source, 1.0) * normalized
            result.rank_by_source[source] = rank
            result.raw_score_by_source[source] = item.score
    return sorted(fused.values(), key=lambda item: (-item.score, item.item_id))
