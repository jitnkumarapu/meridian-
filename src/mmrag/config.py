"""Typed YAML configuration with recursive override support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class PathsConfig(BaseModel):
    raw: Path
    processed: Path
    indexes: Path
    reports: Path


class IngestionConfig(BaseModel):
    min_text_chars: int = 40
    chunk_tokens: int = 300
    overlap_tokens: int = 45
    ocr_language: str = "eng"
    max_pdf_pages: int = 2000
    max_upload_mib: int = 100


class EmbeddingConfig(BaseModel):
    text_model: str
    clip_model: str
    clip_pretrained: str
    batch_size: int = 32
    device: str = "auto"
    test_mode: bool = False


class RetrievalConfig(BaseModel):
    top_k: int = 5
    candidate_k: int = 20
    fusion: str = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    diversity_margin: float = 0.1
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_cache: bool = True


class GenerationConfig(BaseModel):
    provider: str = "mock"
    model: str
    temperature: float = 0
    max_tokens: int = 512
    use_vision: bool = False
    base_url: str | None = None
    key_env: str | None = None
    timeout: float = 60
    retry_count: int = 2
    streaming: bool = False


class BaselineConfig(BaseModel):
    tables_flat: bool = False
    figures: bool = True


class AppConfig(BaseModel):
    seed: int = 42
    paths: PathsConfig
    ingestion: IngestionConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    baseline: BaselineConfig


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path = Path("configs/default.yaml"), override: Path | None = None) -> AppConfig:
    repository_root = Path(__file__).resolve().parents[2]
    load_dotenv(repository_root / ".env", override=False)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if override:
        data = _merge(data, yaml.safe_load(override.read_text(encoding="utf-8")))
    return AppConfig.model_validate(data)
