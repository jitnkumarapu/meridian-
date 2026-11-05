"""Application-level orchestration shared by CLI, API, and both user interfaces."""

from __future__ import annotations

import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from mmrag.baseline import to_text_only
from mmrag.config import AppConfig, load_config
from mmrag.embedding import (
    CrossEncoderReranker,
    HashTextEncoder,
    OpenCLIPEncoder,
    SentenceTransformerEncoder,
    TextEncoder,
)
from mmrag.generation import AnswerPipeline, create_llm_client
from mmrag.generation.prompts import prompt_hash
from mmrag.generation.reasoning import reason_financial_question
from mmrag.ingestion import TesseractOCRBackend, ingest_pdf
from mmrag.retrieval import HybridRetriever
from mmrag.retrieval.hybrid import FigureEncoder
from mmrag.schemas import Answer, Chunk, IngestionReport, Modality, RetrievalResult


def load_processed_chunks(processed: Path) -> list[Chunk]:
    """Load validated chunk JSONL files deterministically, rejecting duplicate IDs."""
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for path in sorted(processed.glob("*/chunks.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            chunk = Chunk.model_validate_json(line)
            if chunk.chunk_id in seen:
                raise ValueError(f"duplicate chunk id {chunk.chunk_id!r} in {path}:{number}")
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks


class MMRAGPipeline:
    """One configured corpus with lazily constructed retrievers and answer pipeline."""

    def __init__(self, config: AppConfig, chunks: list[Chunk] | None = None) -> None:
        self.config = config
        self.chunks = chunks if chunks is not None else load_processed_chunks(config.paths.processed)
        self._retrievers: dict[bool, HybridRetriever] = {}
        self._answer_pipeline: AnswerPipeline | None = None

    @property
    def answer_pipeline(self) -> AnswerPipeline:
        """Create the external generator only when an answer actually needs it.

        Ingestion, indexing, retrieval, and deterministic finance answers must not require an API
        credential for a provider that they never call.
        """
        if self._answer_pipeline is None:
            config = self.config.generation
            llm = create_llm_client(
                config.provider,
                config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                base_url=config.base_url,
                key_env=config.key_env,
                timeout=config.timeout,
                retry_count=config.retry_count,
            )
            self._answer_pipeline = AnswerPipeline(llm, use_vision=config.use_vision, provider=config.provider)
        return self._answer_pipeline

    @classmethod
    def from_config(
        cls, config_path: Path = Path("configs/default.yaml"), override: Path | None = None
    ) -> MMRAGPipeline:
        return cls(load_config(config_path, override))

    def _text_encoder(self) -> TextEncoder:
        if self.config.embedding.test_mode:
            return HashTextEncoder()
        device = None if self.config.embedding.device == "auto" else self.config.embedding.device
        return SentenceTransformerEncoder(
            self.config.embedding.text_model,
            device=device,
            batch_size=self.config.embedding.batch_size,
        )

    def _figure_encoder(self) -> FigureEncoder | None:
        if self.config.embedding.test_mode:
            return None
        device = "cpu" if self.config.embedding.device == "auto" else self.config.embedding.device
        return OpenCLIPEncoder(
            self.config.embedding.clip_model,
            pretrained=self.config.embedding.clip_pretrained,
            device=device,
        )

    def _reranker(self) -> CrossEncoderReranker | None:
        if not self.config.retrieval.reranker_enabled:
            return None
        if self.config.embedding.test_mode:
            raise RuntimeError("Reranking requires production encoders; disable it for offline test mode")
        device = None if self.config.embedding.device == "auto" else self.config.embedding.device
        return CrossEncoderReranker(self.config.retrieval.reranker_model, device=device)

    def retriever(self, baseline: bool = False) -> HybridRetriever:
        if baseline not in self._retrievers:
            corpus = to_text_only(self.chunks) if baseline else self.chunks
            if not corpus:
                raise RuntimeError("No processed chunks found. Run `mmrag ingest data/raw` before indexing or asking.")
            text_encoder = self._text_encoder()
            figure_encoder = None if baseline else self._figure_encoder()
            reranker = self._reranker()
            target = self.config.paths.indexes / ("baseline" if baseline else "multimodal")
            persisted = target / "index_metadata.json"
            if persisted.is_file():
                metadata = json.loads(persisted.read_text(encoding="utf-8"))
                if metadata.get("corpus_checksum") == HybridRetriever.corpus_checksum(corpus):
                    self._retrievers[baseline] = HybridRetriever.load(
                        target,
                        text_encoder,
                        figure_encoder=figure_encoder,
                        candidate_k=self.config.retrieval.candidate_k,
                        fusion=self.config.retrieval.fusion,
                        rrf_k=self.config.retrieval.rrf_k,
                        dense_weight=self.config.retrieval.dense_weight,
                        sparse_weight=self.config.retrieval.sparse_weight,
                        reranker=reranker,
                    )
                    return self._retrievers[baseline]
            self._retrievers[baseline] = HybridRetriever(
                corpus,
                text_encoder,
                figure_encoder=figure_encoder,
                candidate_k=self.config.retrieval.candidate_k,
                fusion=self.config.retrieval.fusion,
                rrf_k=self.config.retrieval.rrf_k,
                dense_weight=self.config.retrieval.dense_weight,
                sparse_weight=self.config.retrieval.sparse_weight,
                cache_dir=self.config.paths.indexes / "embedding_cache"
                if self.config.retrieval.embedding_cache
                else None,
                reranker=reranker,
            )
        return self._retrievers[baseline]

    def ingest(self, source: Path, *, force: bool = False) -> list[IngestionReport]:
        sources = sorted(source.glob("*.pdf")) if source.is_dir() else [source]
        if not sources:
            raise ValueError(f"No PDF files found at {source}")
        reports: list[IngestionReport] = []
        try:
            ocr_backend = TesseractOCRBackend()
        except RuntimeError:
            ocr_backend = None
        for path in sources:
            new_chunks, report = ingest_pdf(
                path,
                self.config.paths.processed,
                min_text_chars=self.config.ingestion.min_text_chars,
                chunk_tokens=self.config.ingestion.chunk_tokens,
                overlap_tokens=self.config.ingestion.overlap_tokens,
                ocr_language=self.config.ingestion.ocr_language,
                ocr_backend=ocr_backend,
                force=force,
                max_pages=self.config.ingestion.max_pdf_pages,
            )
            reports.append(report)
            existing = {chunk.chunk_id: chunk for chunk in self.chunks}
            existing.update({chunk.chunk_id: chunk for chunk in new_chunks})
            self.chunks = list(existing.values())
        self._retrievers.clear()
        return reports

    def build_indexes(self, *, baseline: bool = False) -> Path:
        target = self.config.paths.indexes / ("baseline" if baseline else "multimodal")
        self.retriever(baseline).save(target)
        return target

    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        baseline: bool = False,
        modalities: set[Modality] | None = None,
        doc_ids: set[str] | None = None,
        fiscal_years: set[str] | None = None,
    ) -> RetrievalResult:
        if doc_ids is None:
            doc_ids = self._infer_doc_ids(query)
        return self.retriever(baseline).retrieve(
            query,
            k or self.config.retrieval.top_k,
            modalities=modalities,
            doc_ids=doc_ids,
            fiscal_years=fiscal_years,
        )

    def _infer_doc_ids(self, query: str) -> set[str] | None:
        """Route explicit company-name queries to matching corpus documents."""
        query_words = set(re.findall(r"[a-z0-9]+", query.casefold()))
        special_aliases = {
            "alphabet": [{"google"}, {"alphabet", "inc"}],
            "tcs": [{"tata", "consultancy", "services"}],
            "hcltech": [{"hcl", "technologies"}],
            "hdfc-bank": [{"hdfc", "bank"}],
            "asian-paints": [{"asian", "paints"}],
            "itc": [{"itc", "limited"}],
            "reliance": [{"reliance", "industries"}],
            "wipro": [{"wipro", "limited"}],
        }
        matches: set[str] = set()
        for doc_id in {chunk.doc_id for chunk in self.chunks}:
            stem = re.sub(r"-20\d{2}$", "", doc_id.casefold())
            aliases = [set(stem.split("-")), *special_aliases.get(stem, [])]
            if any(alias and alias.issubset(query_words) for alias in aliases):
                matches.add(doc_id)
        return matches or None

    def ask(
        self,
        query: str,
        *,
        k: int | None = None,
        baseline: bool = False,
        modalities: set[Modality] | None = None,
        doc_ids: set[str] | None = None,
        fiscal_years: set[str] | None = None,
        style: str = "compact",
    ) -> Answer:
        result = self.retrieve(
            query, k=k, baseline=baseline, modalities=modalities, doc_ids=doc_ids, fiscal_years=fiscal_years
        )
        deterministic = reason_financial_question(query, result.evidence)
        if deterministic is not None:
            return Answer(
                text=deterministic.text,
                citations=deterministic.citations,
                evidence=result.evidence,
                grounded=True,
                latency_ms=result.latency_ms,
                token_usage=None,
                model_id="deterministic-finance-v1",
                prompt_hash=prompt_hash(f"deterministic-finance-v1:{query}:{deterministic.trace.model_dump_json()}"),
                provider="deterministic",
                style=style,
                calculation=deterministic.trace,
                metadata={"grounding": "deterministic", "evidence_count": len(result.evidence)},
            )
        return self.answer_pipeline.answer(query, result.evidence, style=style)

    def documents(self) -> list[dict[str, object]]:
        manifest: dict[str, dict[str, str]] = {}
        manifest_path = Path("data/manifests/corpus.csv")
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8", newline="") as handle:
                manifest = {row["doc_id"]: row for row in csv.DictReader(handle)}
        reports: list[dict[str, object]] = []
        for path in sorted(self.config.paths.processed.glob("*/report.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            doc_id = str(raw.get("doc_id", path.parent.name))
            source = Path(str(raw.pop("source_path", "")))
            details = manifest.get(doc_id, {})
            raw.update(
                {
                    "filename": source.name,
                    "company": details.get("company", ""),
                    "fiscal_year": details.get("fy", ""),
                    "processing_state": "needs_review" if raw.get("warnings") else "ready",
                    "warning_count": len(raw.get("warnings", [])),
                    "last_ingestion_time": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                    "last_index_time": self._last_index_time(),
                }
            )
            reports.append(raw)
        return reports

    def _last_index_time(self) -> str | None:
        metadata = self.config.paths.indexes / "multimodal" / "index_metadata.json"
        if not metadata.is_file():
            return None
        value = json.loads(metadata.read_text(encoding="utf-8")).get("built_at")
        return str(value) if value else None

    def chunk(self, chunk_id: str) -> Chunk | None:
        return next((chunk for chunk in self.chunks if chunk.chunk_id == chunk_id), None)

    def remove_document_from_index(self, doc_id: str) -> None:
        self.chunks = [chunk for chunk in self.chunks if chunk.doc_id != doc_id]
        self._retrievers.clear()
