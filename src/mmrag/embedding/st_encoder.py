"""Text encoders for production and deterministic offline tests."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import re
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
_TOKEN_RE = re.compile(r"(?u)\b[\w.-]+\b")


@runtime_checkable
class TextEncoder(Protocol):
    """Minimal contract required by the index and retriever."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: list[str]) -> FloatArray: ...


def _normalise(vectors: NDArray[np.floating]) -> FloatArray:
    result = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    normalized = np.divide(result, norms, out=np.zeros_like(result), where=norms > 0)
    return cast(FloatArray, normalized)


class HashTextEncoder:
    """Deterministic feature-hashing encoder, exclusively for offline tests and demos.

    This is intentionally named and identified as a test encoder so it cannot be confused with the
    configured pretrained production representation model.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return f"test/hash-text-v1/{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str]) -> FloatArray:
        vectors = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.casefold())
            features = tokens + [f"{a}::{b}" for a, b in itertools.pairwise(tokens)]
            for token in features:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                index = value % self._dimension
                sign = 1.0 if value & (1 << 63) else -1.0
                vectors[row, index] += sign
        return _normalise(vectors)


class SentenceTransformerEncoder:
    """Lazy Sentence Transformers adapter with normalized float32 output."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError("Sentence Transformers is not installed; install the 'models' project extra") from exc
        self._model_id = model_name
        self._batch_size = batch_size
        model_type: Any = sentence_transformers.SentenceTransformer
        self._model: Any = model_type(model_name, device=device)
        dimension_getter = getattr(self._model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = self._model.get_sentence_embedding_dimension
        self._dimension = int(dimension_getter())

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str]) -> FloatArray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
