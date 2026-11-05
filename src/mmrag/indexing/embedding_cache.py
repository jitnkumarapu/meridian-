"""Content/model identity cache for deterministic embedding reuse."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


class EmbeddingCache:
    def __init__(self, root: Path, *, model_id: str, kind: str) -> None:
        self.root = root / hashlib.sha256(f"{kind}:{model_id}".encode()).hexdigest()[:20]
        self.model_id = model_id
        self.kind = kind
        self.stats = CacheStats()

    def _key(self, content: bytes) -> str:
        identity = self.model_id.encode() + b"\0" + self.kind.encode() + b"\0" + content
        return hashlib.sha256(identity).hexdigest()

    def get(self, content: bytes) -> NDArray[np.float32] | None:
        path = self.root / f"{self._key(content)}.npy"
        if not path.is_file():
            self.stats.misses += 1
            return None
        vector = np.load(path, allow_pickle=False)
        if vector.ndim != 1:
            path.unlink(missing_ok=True)
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return np.asarray(vector, dtype=np.float32)

    def put(self, content: bytes, vector: NDArray[np.floating]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.save(self.root / f"{self._key(content)}.npy", np.asarray(vector, dtype=np.float32), allow_pickle=False)
        metadata = {"model_id": self.model_id, "kind": self.kind, "schema_version": 1}
        (self.root / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
