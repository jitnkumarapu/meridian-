"""Financial table normalization and explicit-label serialization."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TableQuality:
    score: float
    warnings: tuple[str, ...]
    non_empty_ratio: float
    header_ratio: float


@dataclass(frozen=True)
class TableCandidate:
    grid: list[list[object]]
    bbox: tuple[float, float, float, float]
    extractor: str
    quality: TableQuality


def _clean(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def normalize_grid(grid: Sequence[Sequence[object]]) -> list[list[str]]:
    """Rectangularize a grid and propagate merged header labels horizontally."""
    rows = [[_clean(cell) for cell in row] for row in grid]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    if rows:
        previous = ""
        for index, value in enumerate(rows[0]):
            if value:
                previous = value
            elif previous and index > 0:
                rows[0][index] = previous
    return rows


def serialize_table(grid: Sequence[Sequence[object]], title: str = "", units: str = "", page: int | None = None) -> str:
    """Serialize cells with row and column labels, preserving signs, units, and currency."""
    rows = normalize_grid(grid)
    if len(rows) < 2 or len(rows[0]) < 2:
        return ""
    headers = [header or f"Column {index}" for index, header in enumerate(rows[0])]
    prefix = "; ".join(
        item
        for item in (title.strip(), f"Units: {units.strip()}" if units.strip() else "", f"Page: {page}" if page else "")
        if item
    )
    facts: list[str] = []
    for row_index, row in enumerate(rows[1:], start=1):
        row_label = row[0] or f"Row {row_index}"
        cells = [f"{headers[column]} = {value}" for column, value in enumerate(row[1:], start=1) if value]
        if cells:
            facts.append(f"{row_label} | " + "; ".join(cells))
    body = ". ".join(facts)
    return f"{prefix}. {body}".strip(". ") if prefix else body


def infer_units(text: str) -> str:
    match = re.search(
        r"(?:₹|\$|€|£)?\s*(?:in\s+)?(crores?|lakhs?|millions?|billions?|thousands?)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0).strip() if match else ""


def score_table(grid: Sequence[Sequence[object]]) -> TableQuality:
    """Score extraction structure without inventing semantic correctness."""
    rows = normalize_grid(grid)
    if len(rows) < 2 or len(rows[0]) < 2:
        return TableQuality(0.0, ("table has fewer than two rows or columns",), 0.0, 0.0)
    width = len(rows[0])
    total = len(rows) * width
    non_empty = sum(bool(cell) for row in rows for cell in row)
    non_empty_ratio = non_empty / total
    header_ratio = sum(bool(cell) for cell in rows[0]) / width
    warnings: list[str] = []
    if non_empty_ratio < 0.5:
        warnings.append("many blank cells")
    if header_ratio < 0.75:
        warnings.append("incomplete header")
    repeated_rows = len(rows) - len({tuple(row) for row in rows})
    if repeated_rows:
        warnings.append(f"{repeated_rows} repeated row(s)")
    numeric_rows = sum(any(re.search(r"\d", cell) for cell in row[1:]) for row in rows[1:]) / (len(rows) - 1)
    score = max(0.0, min(1.0, 0.45 * non_empty_ratio + 0.3 * header_ratio + 0.25 * numeric_rows))
    return TableQuality(score, tuple(warnings), non_empty_ratio, header_ratio)


def choose_table(candidates: Sequence[TableCandidate]) -> TableCandidate | None:
    """Choose deterministically by structural score, then extractor name."""
    return max(candidates, key=lambda item: (item.quality.score, item.extractor), default=None)


def pdfplumber_page_tables(page: Any) -> list[TableCandidate]:
    """Extract candidates from an already-open pdfplumber page."""
    candidates: list[TableCandidate] = []
    settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
    }
    for table in page.find_tables(table_settings=settings):
        grid = table.extract()
        if grid:
            raw_bbox = table.bbox
            bbox = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
            candidates.append(TableCandidate(grid, bbox, "pdfplumber", score_table(grid)))
    return candidates


def pdfplumber_tables(source_path: Path, page_index: int) -> list[TableCandidate]:
    """Extract table candidates while preserving the standalone public helper."""
    try:
        import pdfplumber
    except ImportError:
        return []
    with pdfplumber.open(source_path) as document:
        return pdfplumber_page_tables(document.pages[page_index - 1])


def table_context(blocks: Sequence[Sequence[object]], bbox: tuple[float, float, float, float]) -> dict[str, object]:
    """Find nearby source text without inventing a title or units."""
    x0, y0, x1, y1 = bbox
    nearby: list[tuple[float, str, str]] = []
    for block in blocks:
        bx0 = float(str(block[0]))
        by0 = float(str(block[1]))
        bx1 = float(str(block[2]))
        by1 = float(str(block[3]))
        raw_text = block[4]
        horizontal_overlap = max(0.0, min(x1, bx1) - max(x0, bx0))
        if horizontal_overlap <= 0:
            continue
        text = _clean(raw_text)
        if not text:
            continue
        if by1 <= y0:
            nearby.append((y0 - by1, text, "above"))
        elif by0 >= y1:
            nearby.append((by0 - y1, text, "below"))
    nearby.sort(key=lambda item: item[0])
    selected = [item for item in nearby if item[0] <= 120][:4]
    combined = " ".join(item[1] for item in selected)
    title = next(
        (text for _, text, _ in selected if re.search(r"\b(table|statement|results?|performance)\b", text, re.I)),
        "",
    )
    periods = sorted(set(re.findall(r"\b(?:FY\s*)?20\d{2}\b", combined, flags=re.I)))
    return {
        "title": title,
        "units": infer_units(combined),
        "periods": periods,
        "context_sources": [
            {"text": text, "distance": round(distance, 2), "position": position}
            for distance, text, position in selected
        ],
    }
