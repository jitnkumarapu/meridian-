"""Embedded-image extraction with caption/surrounding-text linking."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageStat

from mmrag.schemas import BoundingBox, Chunk, Modality

if TYPE_CHECKING:
    from mmrag.ingestion.ocr import OCRBackend


def _caption(page: Any, bbox: tuple[float, float, float, float]) -> str:
    words = page.get_text("blocks")
    x0, y0, x1, y1 = bbox
    candidates: list[tuple[float, str]] = []
    for block in words:
        bx0, by0, bx1, by1, text = block[:5]
        text = str(text).strip().replace("\n", " ")
        if not text:
            continue
        horizontal_overlap = max(0.0, min(x1, bx1) - max(x0, bx0))
        if horizontal_overlap <= 0:
            continue
        distance = min(abs(by0 - y1), abs(y0 - by1))
        lowered = text.lower()
        if distance <= 100 and any(tag in lowered for tag in ("figure", "chart", "exhibit", "source:")):
            candidates.append((distance, text))
    return min(candidates, default=(0.0, ""), key=lambda item: item[0])[1]


def _surrounding_text(page: Any, bbox: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bbox
    candidates: list[tuple[float, str]] = []
    for block in page.get_text("blocks", sort=True):
        bx0, by0, bx1, by1, raw_text = block[:5]
        text = str(raw_text).strip().replace("\n", " ")
        if not text:
            continue
        horizontal_overlap = max(0.0, min(x1, bx1) - max(x0, bx0))
        if horizontal_overlap <= 0:
            continue
        distance = min(abs(float(by0) - y1), abs(y0 - float(by1)))
        if distance <= 180:
            candidates.append((distance, text))
    return " ".join(text for _, text in sorted(candidates)[:3])


def _record_decisions(output_dir: Path, decisions: list[dict[str, object]]) -> None:
    if not decisions:
        return
    path = output_dir.parent / "figure_review.jsonl"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in decisions)
    path.write_text(existing + payload, encoding="utf-8")


def extract_figures(
    document: Any,
    page: Any,
    doc_id: str,
    page_number: int,
    source_path: Path,
    output_dir: Path,
    min_width: int = 160,
    min_height: int = 100,
    scanned_page: bool = False,
    ocr_backend: OCRBackend | None = None,
    ocr_language: str = "eng",
    image_page_counts: dict[int, int] | None = None,
    max_vector_area: float = 0.95,
) -> list[Chunk]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    decisions: list[dict[str, object]] = []
    if scanned_page:
        decisions.append({"page": page_number, "decision": "rejected", "reason": "scanned_page_ocr_first"})
        _record_decisions(output_dir, decisions)
        return chunks
    seen: set[int] = set()
    for image in page.get_images(full=True):
        xref = int(image[0])
        if xref in seen:
            continue
        seen.add(xref)
        info = document.extract_image(xref)
        width, height = int(info.get("width", 0)), int(info.get("height", 0))
        if width < min_width or height < min_height:
            decisions.append({"page": page_number, "xref": xref, "decision": "rejected", "reason": "tiny_image"})
            continue
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        rect = rects[0]
        page_rect = page.rect
        area_ratio = (float(rect.width) * float(rect.height)) / max(float(page_rect.width * page_rect.height), 1.0)
        if area_ratio < 0.02:
            decisions.append({"page": page_number, "xref": xref, "decision": "rejected", "reason": "small_area"})
            continue
        position = float(rect.y0) / max(float(page_rect.height), 1.0)
        repeated_pages = (
            image_page_counts.get(xref, 1)
            if image_page_counts is not None
            else sum(bool(document[index].get_image_rects(xref)) for index in range(document.page_count))
        )
        if repeated_pages >= max(3, document.page_count // 3) and (position < 0.12 or position > 0.82):
            decisions.append(
                {"page": page_number, "xref": xref, "decision": "rejected", "reason": "repeated_header_footer"}
            )
            continue
        try:
            image = Image.open(io.BytesIO(info["image"])).convert("L").resize((32, 32))
            variance = float(ImageStat.Stat(image).var[0])
        except Exception:
            variance = 100.0
        if variance < 2.0:
            decisions.append({"page": page_number, "xref": xref, "decision": "rejected", "reason": "near_blank"})
            continue
        bbox_tuple = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        digest = hashlib.sha256(info["image"]).hexdigest()[:12]
        chunk_id = f"{doc_id}-p{page_number}-figure-{digest}"
        image_path = output_dir / f"{chunk_id}.{info.get('ext', 'png')}"
        image_path.write_bytes(info["image"])
        caption = _caption(page, bbox_tuple)
        surrounding = _surrounding_text(page, bbox_tuple)
        figure_ocr = ""
        ocr_confidence: float | None = None
        if ocr_backend is not None:
            figure_ocr, ocr_confidence = ocr_backend.extract(info["image"], ocr_language)
        if not caption and not surrounding and variance > 5000 and area_ratio > 0.15:
            decisions.append({"page": page_number, "xref": xref, "decision": "rejected", "reason": "decorative_photo"})
            continue
        decisions.append({"page": page_number, "xref": xref, "chunk_id": chunk_id, "decision": "accepted"})
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page=page_number,
                modality=Modality.FIGURE,
                content="\n".join(part for part in (caption, figure_ocr, surrounding) if part),
                caption=caption,
                image_path=image_path,
                source_path=source_path,
                bbox=BoundingBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1),
                metadata={
                    "extraction_method": "embedded_image",
                    "quality_score": round(min(1.0, 0.4 + area_ratio + min(variance / 1000, 0.4)), 4),
                    "decision": "accepted",
                    "ocr_text": figure_ocr,
                    "ocr_confidence": ocr_confidence,
                    "surrounding_text": surrounding,
                    "repeated_pages": repeated_pages,
                    "area_ratio": round(area_ratio, 4),
                    "pixel_variance": round(variance, 4),
                },
            )
        )
    drawings = page.get_drawings()
    if len(drawings) >= 8:
        drawing_rects = [drawing["rect"] for drawing in drawings if drawing.get("rect") is not None]
        if drawing_rects:
            vector_bbox = (
                max(float(page.rect.x0), min(float(rect.x0) for rect in drawing_rects)),
                max(float(page.rect.y0), min(float(rect.y0) for rect in drawing_rects)),
                min(float(page.rect.x1), max(float(rect.x1) for rect in drawing_rects)),
                min(float(page.rect.y1), max(float(rect.y1) for rect in drawing_rects)),
            )
            vx0, vy0, vx1, vy1 = vector_bbox
            vector_area = ((vx1 - vx0) * (vy1 - vy0)) / max(float(page.rect.width * page.rect.height), 1.0)
            caption = _caption(page, vector_bbox)
            context = _surrounding_text(page, vector_bbox)
            overlaps_raster = any(
                chunk.bbox is not None
                and max(0.0, min(vx1, chunk.bbox.x1) - max(vx0, chunk.bbox.x0))
                * max(0.0, min(vy1, chunk.bbox.y1) - max(vy0, chunk.bbox.y0))
                / max((vx1 - vx0) * (vy1 - vy0), 1.0)
                > 0.5
                for chunk in chunks
            )
            chart_context = any(tag in f"{caption} {context}".casefold() for tag in ("chart", "figure", "exhibit"))
            if (
                0.04 <= vector_area <= max_vector_area
                and not overlaps_raster
                and (chart_context or len(drawings) >= 20)
            ):
                clip = page.rect.__class__(vx0, vy0, vx1, vy1)
                payload = page.get_pixmap(clip=clip, dpi=160, alpha=False).tobytes("png")
                digest = hashlib.sha256(payload).hexdigest()[:12]
                chunk_id = f"{doc_id}-p{page_number}-figure-vector-{digest}"
                image_path = output_dir / f"{chunk_id}.png"
                image_path.write_bytes(payload)
                vector_ocr = ""
                if ocr_backend is not None:
                    vector_ocr, _ = ocr_backend.extract(payload, ocr_language)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        page=page_number,
                        modality=Modality.FIGURE,
                        content="\n".join(part for part in (caption, vector_ocr, context) if part),
                        caption=caption,
                        image_path=image_path,
                        source_path=source_path,
                        bbox=BoundingBox(x0=vx0, y0=vy0, x1=vx1, y1=vy1),
                        metadata={
                            "extraction_method": "vector_region",
                            "quality_score": round(min(1.0, 0.5 + vector_area), 4),
                            "decision": "accepted",
                            "ocr_text": vector_ocr,
                            "surrounding_text": context,
                        },
                    )
                )
                decisions.append({"page": page_number, "chunk_id": chunk_id, "decision": "accepted"})
    _record_decisions(output_dir, decisions)
    return chunks
