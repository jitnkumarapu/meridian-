"""Optional pretrained cross-encoder adapter."""

from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model_id: str, *, device: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install the 'models' extra to enable reranking") from exc
        self.model_id = model_id
        self._model = CrossEncoder(model_id, device=device)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        values = self._model.predict([(query, document) for document in documents])
        return [float(value) for value in values]
