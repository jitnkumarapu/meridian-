"""Portable exact inner-product index over normalized vectors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScoredItem:
    item_id: str
    score: float


class DenseStore:
    """Exact search suitable for tens-of-documents scale, with safe persistence."""

    def __init__(
        self,
        item_ids: list[str],
        vectors: NDArray[np.floating],
        *,
        model_id: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(item_ids):
            raise ValueError("vectors must be a 2D array with one row per item id")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("item ids must be unique")
        self.item_ids = list(item_ids)
        self.vectors = array
        self.model_id = model_id
        self.metadata = dict(metadata or {})

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    def search(self, query: NDArray[np.floating], k: int, *, allowed_ids: set[str] | None = None) -> list[ScoredItem]:
        if k <= 0 or not self.item_ids:
            return []
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise ValueError(f"query dimension {vector.shape[0]} != index dimension {self.dimension}")
        scores = self.vectors @ vector
        candidates = [
            (index, float(score))
            for index, score in enumerate(scores)
            if allowed_ids is None or self.item_ids[index] in allowed_ids
        ]
        candidates.sort(key=lambda pair: (-pair[1], self.item_ids[pair[0]]))
        return [ScoredItem(self.item_ids[i], score) for i, score in candidates[:k]]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors, allow_pickle=False)
        checksum = hashlib.sha256(self.vectors.tobytes()).hexdigest()
        metadata = {
            "schema_version": 2,
            "model_id": self.model_id,
            "item_ids": self.item_ids,
            "dimension": self.dimension,
            "normalization": "l2",
            "vector_checksum": checksum,
            **self.metadata,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, *, expected_model_id: str | None = None) -> DenseStore:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        model_id = str(metadata["model_id"])
        if expected_model_id is not None and model_id != expected_model_id:
            raise ValueError(f"index uses {model_id!r}, expected {expected_model_id!r}")
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        expected_dimension = metadata.get("dimension")
        if expected_dimension is not None and int(expected_dimension) != vectors.shape[1]:
            raise ValueError("index metadata dimension does not match stored vectors")
        expected_checksum = metadata.get("vector_checksum")
        actual_checksum = hashlib.sha256(vectors.tobytes()).hexdigest()
        if expected_checksum is not None and expected_checksum != actual_checksum:
            raise ValueError("index vector checksum mismatch")
        reserved = {"schema_version", "model_id", "item_ids", "dimension", "normalization", "vector_checksum"}
        extra = {key: value for key, value in metadata.items() if key not in reserved}
        return cls(list(metadata["item_ids"]), vectors, model_id=model_id, metadata=extra)
