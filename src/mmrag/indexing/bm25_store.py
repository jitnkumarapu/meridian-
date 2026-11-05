"""BM25 with finance-aware tokenization and JSON persistence."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .dense_store import ScoredItem

_FINANCE_TOKEN = re.compile(
    r"FY\d{2,4}|[₹$€£](?:-?\(?\d[\d,]*(?:\.\d+)?\)?%?)|"
    r"-?\(?\d[\d,]*(?:\.\d+)?\)?%?|[A-Za-z]+(?:[-'][A-Za-z0-9]+)*",
    re.IGNORECASE,
)


def financial_tokenize(text: str) -> list[str]:
    """Tokenize without destroying fiscal years, currency, commas, signs, or percentages."""

    return [token.casefold() for token in _FINANCE_TOKEN.findall(text)]


class BM25Store:
    def __init__(
        self,
        item_ids: list[str],
        documents: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(item_ids) != len(documents):
            raise ValueError("documents must contain one entry per item id")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("item ids must be unique")
        self.item_ids = list(item_ids)
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._tokens = [financial_tokenize(document) for document in documents]
        self._term_counts = [Counter(tokens) for tokens in self._tokens]
        self._avgdl = sum(map(len, self._tokens)) / len(self._tokens) if self._tokens else 0.0
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        count = len(self._tokens)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, k: int, *, allowed_ids: set[str] | None = None) -> list[ScoredItem]:
        if k <= 0:
            return []
        query_terms = financial_tokenize(query)
        candidates: list[ScoredItem] = []
        for item_id, counts, tokens in zip(self.item_ids, self._term_counts, self._tokens, strict=True):
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            score = 0.0
            length_ratio = len(tokens) / self._avgdl if self._avgdl else 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += self._idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            candidates.append(ScoredItem(item_id, score))
        candidates.sort(key=lambda item: (-item.score, item.item_id))
        return candidates[:k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "item_ids": self.item_ids,
                    "documents": self.documents,
                    "k1": self.k1,
                    "b": self.b,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> BM25Store:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["item_ids"], data["documents"], k1=data["k1"], b=data["b"])
