"""OCR only sparse pages recorded by completed ingestion reports and merge text chunks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from mmrag.extraction.text import chunk_text
from mmrag.ingestion.ocr import TesseractOCRBackend
from mmrag.schemas import Chunk, Modality

SPARSE_WARNING = re.compile(r"page (\d+): sparse text; OCR backend unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/real_processed"))
    parser.add_argument("--language", default="eng")
    args = parser.parse_args()
    backend = TesseractOCRBackend()
    import pymupdf

    total_pages = 0
    for doc_dir in sorted(path for path in args.processed.iterdir() if path.is_dir()):
        report_path = doc_dir / "report.json"
        chunks_path = doc_dir / "chunks.jsonl"
        if not report_path.is_file() or not chunks_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sparse_pages = {
            int(match.group(1))
            for warning in report.get("warnings", [])
            if (match := SPARSE_WARNING.fullmatch(warning)) is not None
        }
        if not sparse_pages:
            continue
        source = args.raw / f"{doc_dir.name}.pdf"
        document: Any = pymupdf.open(source)
        chunks = [Chunk.model_validate_json(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        succeeded: set[int] = set()
        classified: dict[int, str] = {}
        for page_number in sorted(sparse_pages):
            page = document[page_number - 1]
            payload = page.get_pixmap(dpi=200, alpha=False).tobytes("png")
            text, confidence = backend.extract(payload, args.language)
            if not text.strip():
                if page.get_images(full=True) or page.get_drawings():
                    classified[page_number] = f"page {page_number}: image-only decorative page; OCR returned no text"
                else:
                    classified[page_number] = f"page {page_number}: intentionally blank page"
                continue
            for index, content in enumerate(chunk_text(text, 300, 45), 1):
                chunk = Chunk(
                    chunk_id=f"{doc_dir.name}-p{page_number}-text-{index}",
                    doc_id=doc_dir.name,
                    page=page_number,
                    modality=Modality.TEXT,
                    content=content,
                    source_path=source.resolve(),
                    ocr_used=True,
                    metadata={"ocr_language": args.language, "ocr_confidence": confidence},
                )
                by_id[chunk.chunk_id] = chunk
            succeeded.add(page_number)
        document.close()
        merged = sorted(by_id.values(), key=lambda chunk: chunk.chunk_id)
        chunks_path.write_text("\n".join(chunk.model_dump_json() for chunk in merged), encoding="utf-8")
        report["ocr_pages"] = int(report.get("ocr_pages", 0)) + len(succeeded)
        report["text_chunks"] = sum(chunk.modality is Modality.TEXT for chunk in merged)
        retained_warnings: list[str] = []
        for warning in report.get("warnings", []):
            match = SPARSE_WARNING.fullmatch(warning)
            if match is None:
                retained_warnings.append(warning)
                continue
            page_number = int(match.group(1))
            if page_number in classified:
                retained_warnings.append(classified[page_number])
            elif page_number not in succeeded:
                retained_warnings.append(warning)
        report["warnings"] = retained_warnings
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        total_pages += len(succeeded)
        print(f"{doc_dir.name}: ocr_pages={sorted(succeeded)}")
    print(f"ocr_augmented_pages={total_pages}")


if __name__ == "__main__":
    main()
