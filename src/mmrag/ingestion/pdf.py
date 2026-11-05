"""Idempotent PDF ingestion into JSONL chunks and a typed report."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, cast

from mmrag.extraction.figures import extract_figures
from mmrag.extraction.tables import (
    TableCandidate,
    choose_table,
    normalize_grid,
    pdfplumber_page_tables,
    pdfplumber_tables,
    score_table,
    serialize_table,
    table_context,
)
from mmrag.extraction.text import chunk_text, nearest_heading
from mmrag.finance import FinanceError, parse_financial_value
from mmrag.ingestion.ocr import OCRBackend
from mmrag.schemas import BoundingBox, Chunk, IngestionReport, Modality

LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _doc_id(path: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return slug or "document"


def _page_png(page: Any) -> bytes:
    matrix = page.parent.Matrix(2, 2) if hasattr(page.parent, "Matrix") else None
    pixmap = page.get_pixmap(matrix=matrix, alpha=False) if matrix else page.get_pixmap(dpi=200, alpha=False)
    return cast(bytes, pixmap.tobytes("png"))


def _likely_borderless_table(text: str) -> bool:
    lowered = text.casefold()
    financial_header = any(
        phrase in lowered for phrase in ("year ended", "as at", "revenue", "net sales", "financial performance")
    )
    numeric_cells = len(re.findall(r"(?<!\w)[(₹$€£]?\d[\d,.]*(?:%|\))?", text))
    return financial_header and numeric_cells >= 8


def ingest_pdf(
    source_path: Path,
    output_root: Path,
    *,
    min_text_chars: int = 40,
    chunk_tokens: int = 300,
    overlap_tokens: int = 45,
    ocr_language: str = "eng",
    ocr_backend: OCRBackend | None = None,
    force: bool = False,
    max_pages: int = 2000,
) -> tuple[list[Chunk], IngestionReport]:
    """Ingest one PDF; corrupt/encrypted documents return a warning report, not a batch crash."""
    import pymupdf

    source_path = source_path.resolve()
    checksum = _sha256(source_path)
    doc_id = _doc_id(source_path)
    doc_dir = output_root / doc_id
    chunks_path = doc_dir / "chunks.jsonl"
    report_path = doc_dir / "report.json"
    if not force and chunks_path.exists() and report_path.exists():
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        if report_data.get("sha256") == checksum:
            cached_chunks = [
                Chunk.model_validate_json(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()
            ]
            return cached_chunks, IngestionReport.model_validate(report_data)

    report = IngestionReport(doc_id=doc_id, source_path=source_path, sha256=checksum)
    chunks: list[Chunk] = []
    try:
        document: Any = pymupdf.open(source_path)  # type: ignore[no-untyped-call]
        if document.needs_pass:
            report.warnings.append("encrypted PDF skipped")
            document.close()
            return chunks, report
        if document.page_count > max_pages:
            report.warnings.append(f"PDF has {document.page_count} pages; limit is {max_pages}")
            document.close()
            return chunks, report
        report.pages = document.page_count
        image_page_counts: dict[int, int] = {}
        for image_page in document:
            for xref in {int(image[0]) for image in image_page.get_images(full=True)}:
                image_page_counts[xref] = image_page_counts.get(xref, 0) + 1
        try:
            import pdfplumber

            plumber_document: Any | None = pdfplumber.open(source_path)
        except ImportError:
            plumber_document = None
        figures_dir = doc_dir / "figures"
        for page_offset in range(document.page_count):
            page_index = page_offset + 1
            page: Any = document[page_offset]
            raw_blocks = page.get_text("blocks", sort=True)
            block_text = [str(block[4]).strip() for block in raw_blocks if str(block[4]).strip()]
            text = "\n".join(block_text)
            ocr_used = False
            ocr_confidence: float | None = None
            if len(text.strip()) < min_text_chars:
                if ocr_backend is None:
                    report.warnings.append(f"page {page_index}: sparse text; OCR backend unavailable")
                else:
                    text, ocr_confidence = ocr_backend.extract(_page_png(page), ocr_language)
                    ocr_used = True
                    report.ocr_pages += 1
            heading = nearest_heading(block_text)
            for index, content in enumerate(chunk_text(text, chunk_tokens, overlap_tokens), start=1):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}-p{page_index}-text-{index}",
                        doc_id=doc_id,
                        page=page_index,
                        modality=Modality.TEXT,
                        content=content,
                        source_path=source_path,
                        section_heading=heading,
                        ocr_used=ocr_used,
                        metadata={"ocr_language": ocr_language, "ocr_confidence": ocr_confidence},
                    )
                )

            try:
                found = page.find_tables()
                pymupdf_candidates = [
                    TableCandidate(
                        table.extract(),
                        (
                            float(table.bbox[0]),
                            float(table.bbox[1]),
                            float(table.bbox[2]),
                            float(table.bbox[3]),
                        ),
                        "pymupdf",
                        score_table(table.extract()),
                    )
                    for table in found.tables
                    if table.extract()
                ]
                if not pymupdf_candidates and _likely_borderless_table(text):
                    text_tables = page.find_tables(vertical_strategy="text", horizontal_strategy="text")
                    pymupdf_candidates = [
                        TableCandidate(
                            table.extract(),
                            (
                                float(table.bbox[0]),
                                float(table.bbox[1]),
                                float(table.bbox[2]),
                                float(table.bbox[3]),
                            ),
                            "pymupdf-text",
                            score_table(table.extract()),
                        )
                        for table in text_tables.tables
                        if table.extract()
                    ]
                plumber_candidates = []
                if pymupdf_candidates:
                    plumber_candidates = (
                        pdfplumber_page_tables(plumber_document.pages[page_offset])
                        if plumber_document is not None
                        else pdfplumber_tables(source_path, page_index)
                    )
                candidate_count = max(len(pymupdf_candidates), len(plumber_candidates))
                for table_index in range(candidate_count):
                    alternatives = [
                        candidates[table_index]
                        for candidates in (pymupdf_candidates, plumber_candidates)
                        if table_index < len(candidates)
                    ]
                    selected = choose_table(alternatives)
                    if selected is None:
                        continue
                    grid = normalize_grid(selected.grid)
                    context = table_context(raw_blocks, selected.bbox)
                    title = str(context["title"])
                    units = str(context["units"])
                    serialized = serialize_table(grid, title=title, units=units, page=page_index)
                    if serialized:
                        chunk_id = f"{doc_id}-p{page_index}-table-{table_index + 1}"
                        rect = selected.bbox
                        chunk = Chunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            page=page_index,
                            modality=Modality.TABLE,
                            content=serialized,
                            source_path=source_path,
                            bbox=BoundingBox(x0=rect[0], y0=rect[1], x1=rect[2], y1=rect[3]),
                            caption=title,
                            raw_grid=grid,
                            metadata={
                                "units": units,
                                "periods": context["periods"],
                                "context_sources": context["context_sources"],
                                "extractor": selected.extractor,
                                "candidate_scores": {item.extractor: item.quality.score for item in alternatives},
                                "quality_score": selected.quality.score,
                                "quality_warnings": list(selected.quality.warnings),
                                "display_grid_preserved": True,
                            },
                        )
                        normalized_values: list[dict[str, object]] = []
                        for row in grid[1:]:
                            for column, display in enumerate(row[1:], 1):
                                if not display:
                                    continue
                                try:
                                    value = parse_financial_value(
                                        display,
                                        chunk,
                                        row_label=row[0],
                                        column_label=grid[0][column],
                                        default_scale=next(
                                            (
                                                scale
                                                for scale in ("thousand", "lakh", "million", "crore", "billion")
                                                if scale in units.casefold()
                                            ),
                                            None,
                                        ),
                                    )
                                except (FinanceError, KeyError):
                                    continue
                                normalized_values.append(value.model_dump(mode="json"))
                        chunk.metadata["normalized_values"] = normalized_values
                        review_dir = doc_dir / "table_review"
                        review_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = review_dir / f"{chunk_id}.png"
                        pixmap = page.get_pixmap(clip=pymupdf.Rect(*rect), dpi=150, alpha=False)  # type: ignore[no-untyped-call]
                        crop_path.write_bytes(pixmap.tobytes("png"))
                        chunk.metadata["review_crop"] = str(crop_path)
                        chunks.append(chunk)
            except Exception as error:  # extractor failure must not discard other page content
                report.warnings.append(f"page {page_index}: table extraction failed: {error}")

            try:
                chunks.extend(
                    extract_figures(
                        document,
                        page,
                        doc_id,
                        page_index,
                        source_path,
                        figures_dir,
                        scanned_page=ocr_used,
                        ocr_backend=ocr_backend,
                        ocr_language=ocr_language,
                        image_page_counts=image_page_counts,
                    )
                )
            except Exception as error:
                report.warnings.append(f"page {page_index}: figure extraction failed: {error}")
        if plumber_document is not None:
            plumber_document.close()
        document.close()
    except Exception as error:
        report.warnings.append(f"document skipped: {type(error).__name__}: {error}")

    report.text_chunks = sum(chunk.modality is Modality.TEXT for chunk in chunks)
    report.tables = sum(chunk.modality is Modality.TABLE for chunk in chunks)
    report.figures = sum(chunk.modality is Modality.FIGURE for chunk in chunks)
    doc_dir.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text("\n".join(chunk.model_dump_json() for chunk in chunks), encoding="utf-8")
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    LOGGER.info("ingested %s: %s", source_path, report.model_dump())
    return chunks, report
