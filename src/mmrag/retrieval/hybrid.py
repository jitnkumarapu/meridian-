"""Single public entry point for sparse, text-dense, and figure-dense retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from mmrag.embedding.st_encoder import TextEncoder
from mmrag.indexing import BM25Store, DenseStore, EmbeddingCache
from mmrag.retrieval.query import normalize_query
from mmrag.schemas import Chunk, Evidence, Modality, RetrievalExplanation, RetrievalResult

from .fusion import FusionItem, reciprocal_rank_fusion, weighted_rank_fusion


class FigureEncoder(Protocol):
    @property
    def model_id(self) -> str: ...

    def encode_texts(self, texts: list[str]): ...  # type: ignore[no-untyped-def]

    def encode_images(self, paths: list[Path]): ...  # type: ignore[no-untyped-def]


class Reranker(Protocol):
    model_id: str

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class HybridRetriever:
    """Retrieve chunks while keeping MiniLM, CLIP, and BM25 score spaces separate."""

    def __init__(
        self,
        chunks: list[Chunk],
        text_encoder: TextEncoder,
        *,
        figure_encoder: FigureEncoder | None = None,
        candidate_k: int = 20,
        fusion: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        cache_dir: Path | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk ids must be unique")
        if fusion not in {"rrf", "weighted"}:
            raise ValueError("fusion must be 'rrf' or 'weighted'")
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.text_encoder = text_encoder
        self.figure_encoder = figure_encoder
        self.candidate_k = candidate_k
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reranker = reranker
        self.reranker_scores: dict[str, float] = {}
        self.cache_stats = {"text_hits": 0, "text_misses": 0, "figure_hits": 0, "figure_misses": 0}
        ids = [chunk.chunk_id for chunk in chunks]
        searchable_text = [self._search_text(chunk) for chunk in chunks]
        vectors = self._encode_texts(searchable_text, cache_dir)
        self.text_store = DenseStore(ids, vectors, model_id=text_encoder.model_id)
        self.sparse_store = BM25Store(ids, searchable_text)
        figure_chunks = [chunk for chunk in chunks if chunk.modality is Modality.FIGURE and chunk.image_path]
        self.figure_store: DenseStore | None = None
        if figure_encoder is not None and figure_chunks:
            figure_paths = [chunk.image_path for chunk in figure_chunks if chunk.image_path]
            figure_vectors = self._encode_figures(figure_paths, cache_dir)
            self.figure_store = DenseStore(
                [chunk.chunk_id for chunk in figure_chunks],
                figure_vectors,
                model_id=figure_encoder.model_id,
            )

    def _encode_texts(self, texts: list[str], cache_dir: Path | None) -> np.ndarray:
        if cache_dir is None:
            return np.asarray(self.text_encoder.encode(texts), dtype=np.float32)
        cache = EmbeddingCache(cache_dir, model_id=self.text_encoder.model_id, kind="text")
        vectors: list[np.ndarray | None] = [cache.get(text.encode("utf-8")) for text in texts]
        missing = [index for index, vector in enumerate(vectors) if vector is None]
        if missing:
            encoded = self.text_encoder.encode([texts[index] for index in missing])
            for index, vector in zip(missing, encoded, strict=True):
                vectors[index] = np.asarray(vector, dtype=np.float32)
                cache.put(texts[index].encode("utf-8"), vector)
        self.cache_stats["text_hits"] = cache.stats.hits
        self.cache_stats["text_misses"] = cache.stats.misses
        return np.stack([vector for vector in vectors if vector is not None])

    def _encode_figures(self, paths: list[Path], cache_dir: Path | None) -> np.ndarray:
        assert self.figure_encoder is not None
        if cache_dir is None:
            return np.asarray(self.figure_encoder.encode_images(paths), dtype=np.float32)
        cache = EmbeddingCache(cache_dir, model_id=self.figure_encoder.model_id, kind="image")
        payloads = [path.read_bytes() for path in paths]
        vectors: list[np.ndarray | None] = [cache.get(payload) for payload in payloads]
        missing = [index for index, vector in enumerate(vectors) if vector is None]
        if missing:
            encoded = self.figure_encoder.encode_images([paths[index] for index in missing])
            for index, vector in zip(missing, encoded, strict=True):
                vectors[index] = np.asarray(vector, dtype=np.float32)
                cache.put(payloads[index], vector)
        self.cache_stats["figure_hits"] = cache.stats.hits
        self.cache_stats["figure_misses"] = cache.stats.misses
        return np.stack([vector for vector in vectors if vector is not None])

    @staticmethod
    def _search_text(chunk: Chunk) -> str:
        parts = [chunk.section_heading, chunk.caption, chunk.content]
        return "\n".join(part for part in parts if part)

    def _allowed_ids(
        self, modalities: set[Modality] | None, doc_ids: set[str] | None, fiscal_years: set[str] | None
    ) -> set[str] | None:
        if modalities is None and doc_ids is None and fiscal_years is None:
            return None
        return {
            chunk_id
            for chunk_id, chunk in self.chunks.items()
            if (modalities is None or chunk.modality in modalities)
            and (doc_ids is None or chunk.doc_id in doc_ids)
            and (
                fiscal_years is None
                or any(year.casefold() in self._search_text(chunk).casefold() for year in fiscal_years)
            )
        }

    def retrieve(
        self,
        query: str,
        k: int,
        modalities: set[Modality] | None = None,
        doc_ids: set[str] | None = None,
        fiscal_years: set[str] | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        allowed = self._allowed_ids(modalities, doc_ids, fiscal_years)
        normalized = normalize_query(query)
        text_query = self.text_encoder.encode([normalized.normalized])[0]
        rankings = {
            "text_dense": self.text_store.search(text_query, self.candidate_k, allowed_ids=allowed),
            "bm25": self.sparse_store.search(normalized.normalized, self.candidate_k, allowed_ids=allowed),
        }
        if self.figure_store is not None and self.figure_encoder is not None and self._is_visual_query(query):
            clip_query = self.figure_encoder.encode_texts([normalized.normalized])[0]
            rankings["figure_dense"] = self.figure_store.search(clip_query, self.candidate_k, allowed_ids=allowed)
        if self.fusion == "rrf":
            fused = reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k)
        else:
            fused = weighted_rank_fusion(
                rankings,
                weights={
                    "text_dense": self.dense_weight,
                    "figure_dense": self.dense_weight,
                    "bm25": self.sparse_weight,
                },
            )
        self.reranker_scores = {}
        if self.reranker is not None:
            eligible_positions = [
                index for index, item in enumerate(fused) if self.chunks[item.item_id].modality is not Modality.FIGURE
            ]
            documents = [self._search_text(self.chunks[fused[index].item_id]) for index in eligible_positions]
            scores = self.reranker.score(query, documents)
            if len(scores) != len(eligible_positions):
                raise ValueError("reranker returned an unexpected number of scores")
            ranked = sorted(
                zip((fused[index] for index in eligible_positions), scores, strict=True),
                key=lambda pair: (-pair[1], pair[0].item_id),
            )
            for position, (item, score) in zip(eligible_positions, ranked, strict=True):
                fused[position] = item
                self.reranker_scores[item.item_id] = score
        filters = {
            "modalities": sorted(item.value for item in modalities or set()),
            "documents": sorted(doc_ids or set()),
            "fiscal_years": sorted(fiscal_years or set()),
        }
        evidence = [self._evidence(item, rank=index, filters=filters) for index, item in enumerate(fused[:k], 1)]
        return RetrievalResult(
            query=query,
            normalized_query=normalized.normalized,
            expansions=list(normalized.expansions),
            evidence=evidence,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _is_visual_query(query: str) -> bool:
        """Use the image index only when the user asks about visual evidence."""
        return (
            re.search(
                r"\b(?:figure|figures|chart|charts|graph|graphs|graphic|graphics|infographic|infographics|"
                r"visual|visuals|diagram|diagrams|illustration|illustrations|plot|plots|plotted|trend|trends|"
                r"shown|highlighted|distribution)\b",
                query,
                flags=re.IGNORECASE,
            )
            is not None
        )

    def _evidence(self, item: FusionItem, *, rank: int, filters: dict[str, list[str]]) -> Evidence:
        dense_scores = [value for source, value in item.raw_score_by_source.items() if source.endswith("dense")]
        return Evidence(
            chunk=self.chunks[item.item_id],
            dense_score=max(dense_scores) if dense_scores else None,
            bm25_score=item.raw_score_by_source.get("bm25"),
            fused_score=item.score,
            rank_by_source=item.rank_by_source,
            explanation=RetrievalExplanation(
                dense_rank=item.rank_by_source.get("text_dense"),
                dense_score=item.raw_score_by_source.get("text_dense"),
                bm25_rank=item.rank_by_source.get("bm25"),
                bm25_score=item.raw_score_by_source.get("bm25"),
                figure_rank=item.rank_by_source.get("figure_dense"),
                figure_score=item.raw_score_by_source.get("figure_dense"),
                fusion_contribution={
                    source: (1 / (self.rrf_k + source_rank) if self.fusion == "rrf" else 0.0)
                    for source, source_rank in item.rank_by_source.items()
                },
                reranker_score=self.reranker_scores.get(item.item_id),
                filters=filters,
                final_rank=rank,
            ),
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.text_store.save(directory / "text_dense")
        self.sparse_store.save(directory / "bm25.json")
        if self.figure_store is not None:
            self.figure_store.save(directory / "figure_dense")
        (directory / "chunks.json").write_text(
            json.dumps(
                [chunk.model_dump(mode="json") for chunk in self.chunks.values()],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        metadata = {
            "schema_version": 1,
            "text_encoder": self.text_encoder.model_id,
            "text_checkpoint": self.text_encoder.model_id,
            "figure_encoder": self.figure_encoder.model_id if self.figure_encoder else None,
            "figure_checkpoint": self.figure_encoder.model_id if self.figure_encoder else None,
            "reranker": self.reranker.model_id if self.reranker else None,
            "dimension": self.text_store.dimension,
            "normalization": "l2",
            "preprocessing_version": "finance-query-v1",
            "corpus_checksum": self.corpus_checksum(list(self.chunks.values())),
            "configuration": {
                "candidate_k": self.candidate_k,
                "fusion": self.fusion,
                "rrf_k": self.rrf_k,
                "dense_weight": self.dense_weight,
                "sparse_weight": self.sparse_weight,
            },
            "cache": self.cache_stats,
            "built_at": datetime.now(UTC).isoformat(),
        }
        metadata["config_hash"] = hashlib.sha256(
            json.dumps(metadata["configuration"], sort_keys=True).encode()
        ).hexdigest()
        (directory / "index_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def corpus_checksum(chunks: list[Chunk]) -> str:
        """Return the stable checksum used to reject stale persisted indexes."""
        payload = "\n".join(
            f"{chunk.chunk_id}:{chunk.model_dump_json()}" for chunk in sorted(chunks, key=lambda item: item.chunk_id)
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def load(
        cls,
        directory: Path,
        text_encoder: TextEncoder,
        *,
        figure_encoder: FigureEncoder | None = None,
        candidate_k: int = 20,
        fusion: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        reranker: Reranker | None = None,
    ) -> HybridRetriever:
        """Load persisted stores without re-encoding the corpus."""
        expected_config = {
            "candidate_k": candidate_k,
            "fusion": fusion,
            "rrf_k": rrf_k,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
        }
        metadata = cls.validate_index(
            directory,
            text_encoder_id=text_encoder.model_id,
            figure_encoder_id=figure_encoder.model_id if figure_encoder else None,
            expected_config=expected_config,
        )
        chunks = [
            Chunk.model_validate(item) for item in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        ]
        if cls.corpus_checksum(chunks) != metadata.get("corpus_checksum"):
            raise ValueError("persisted index corpus checksum mismatch")

        instance = cls.__new__(cls)
        instance.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        if len(instance.chunks) != len(chunks):
            raise ValueError("persisted index contains duplicate chunk ids")
        instance.text_encoder = text_encoder
        instance.figure_encoder = figure_encoder
        instance.candidate_k = candidate_k
        instance.fusion = fusion
        instance.rrf_k = rrf_k
        instance.dense_weight = dense_weight
        instance.sparse_weight = sparse_weight
        instance.reranker = reranker
        instance.reranker_scores = {}
        raw_cache = metadata.get("cache", {})
        instance.cache_stats = cast(dict[str, int], raw_cache) if isinstance(raw_cache, dict) else {}
        instance.text_store = DenseStore.load(directory / "text_dense", expected_model_id=text_encoder.model_id)
        instance.sparse_store = BM25Store.load(directory / "bm25.json")
        figure_path = directory / "figure_dense"
        instance.figure_store = (
            DenseStore.load(figure_path, expected_model_id=figure_encoder.model_id)
            if figure_encoder is not None and figure_path.is_dir()
            else None
        )
        stored_ids = set(instance.chunks)
        if set(instance.text_store.item_ids) != stored_ids or set(instance.sparse_store.item_ids) != stored_ids:
            raise ValueError("persisted index stores do not match persisted chunks")
        return instance

    @staticmethod
    def validate_index(
        directory: Path,
        *,
        text_encoder_id: str,
        figure_encoder_id: str | None,
        expected_config: dict[str, object] | None = None,
    ) -> dict[str, object]:
        metadata = json.loads((directory / "index_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("text_encoder") != text_encoder_id:
            raise ValueError("text encoder is incompatible with saved index")
        if metadata.get("figure_encoder") != figure_encoder_id:
            raise ValueError("figure encoder is incompatible with saved index")
        if expected_config is not None:
            expected_hash = hashlib.sha256(json.dumps(expected_config, sort_keys=True).encode()).hexdigest()
            if metadata.get("config_hash") != expected_hash:
                raise ValueError("retrieval configuration is incompatible with saved index")
        return cast(dict[str, object], metadata)
