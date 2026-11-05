"""Shared contracts used by ingestion, retrieval, generation, API, and evaluation."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Modality(StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    page: int = Field(ge=1)
    modality: Modality
    content: str
    source_path: Path
    image_path: Path | None = None
    bbox: BoundingBox | None = None
    caption: str = ""
    section_heading: str = ""
    ocr_used: bool = False
    raw_grid: list[list[str]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        return f"[{self.doc_id}, p.{self.page}, {self.modality.value}]"


class Evidence(BaseModel):
    chunk: Chunk
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float
    rank_by_source: dict[str, int] = Field(default_factory=dict)
    explanation: RetrievalExplanation | None = None


class RetrievalExplanation(BaseModel):
    """Auditable contribution of each retrieval stage to one result."""

    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    figure_rank: int | None = None
    figure_score: float | None = None
    fusion_contribution: dict[str, float] = Field(default_factory=dict)
    reranker_score: float | None = None
    filters: dict[str, list[str]] = Field(default_factory=dict)
    final_rank: int | None = None


class CalculationOperand(BaseModel):
    display_text: str
    value: Decimal
    currency: str | None = None
    scale: str | None = None
    unit: str | None = None
    percentage: bool = False
    fiscal_period: str | None = None
    row_label: str | None = None
    column_label: str | None = None
    source_chunk_id: str
    doc_id: str
    page: int = Field(ge=1)
    modality: Modality


class CalculationTrace(BaseModel):
    operation: str
    operands: list[CalculationOperand]
    formula: str
    normalized_unit: str | None = None
    result: Decimal
    citations: list[str]


class Citation(BaseModel):
    doc_id: str
    page: int = Field(ge=1)
    modality: Modality
    chunk_id: str | None = None

    @property
    def label(self) -> str:
        return f"[{self.doc_id}, p.{self.page}, {self.modality.value}]"


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    evidence: list[Evidence]
    grounded: bool
    latency_ms: float = Field(ge=0)
    token_usage: int | None = None
    model_id: str
    prompt_hash: str
    provider: str | None = None
    retries: int = Field(default=0, ge=0)
    style: str = "compact"
    calculation: CalculationTrace | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def grounded_answers_have_citations(self) -> Answer:
        if self.grounded and not self.citations:
            raise ValueError("grounded answers require at least one citation")
        return self


class RetrievalResult(BaseModel):
    query: str
    normalized_query: str | None = None
    expansions: list[str] = Field(default_factory=list)
    evidence: list[Evidence]
    latency_ms: float = Field(ge=0)


class IngestionReport(BaseModel):
    doc_id: str
    source_path: Path
    pages: int = 0
    text_chunks: int = 0
    tables: int = 0
    figures: int = 0
    ocr_pages: int = 0
    warnings: list[str] = Field(default_factory=list)
    sha256: str
