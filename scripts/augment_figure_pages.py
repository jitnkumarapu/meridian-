"""Re-evaluate question-critical visual pages and merge any missing vector figure chunks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mmrag.extraction.figures import extract_figures
from mmrag.schemas import Chunk, Modality

if __package__:
    from scripts.generate_question_draft import SPECS
else:
    from generate_question_draft import SPECS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/real_processed"))
    parser.add_argument("--documents", nargs="*", help="Optional document IDs to augment")
    args = parser.parse_args()
    visual_pages: dict[str, set[int]] = defaultdict(set)
    for spec in SPECS:
        for page, modality, _terms in spec["e"]:
            if modality == "figure":
                visual_pages[spec["doc"]].add(page)

    import pymupdf

    for doc_id, pages in sorted(visual_pages.items()):
        if args.documents and doc_id not in args.documents:
            continue
        source = args.raw / f"{doc_id}.pdf"
        doc_dir = args.processed / doc_id
        chunks_path = doc_dir / "chunks.jsonl"
        report_path = doc_dir / "report.json"
        if not chunks_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"incomplete extraction for {doc_id}")
        chunks = [Chunk.model_validate_json(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        document: Any = pymupdf.open(source)
        image_page_counts: dict[int, int] = {}
        for image_page in document:
            for xref in {int(image[0]) for image in image_page.get_images(full=True)}:
                image_page_counts[xref] = image_page_counts.get(xref, 0) + 1
        added = 0
        for page_number in sorted(pages):
            new_chunks = extract_figures(
                document,
                document[page_number - 1],
                doc_id,
                page_number,
                source.resolve(),
                doc_dir / "figures",
                image_page_counts=image_page_counts,
                max_vector_area=1.05,
            )
            for chunk in new_chunks:
                if chunk.chunk_id not in by_id:
                    by_id[chunk.chunk_id] = chunk
                    added += 1
        document.close()
        merged = sorted(by_id.values(), key=lambda chunk: chunk.chunk_id)
        chunks_path.write_text("\n".join(chunk.model_dump_json() for chunk in merged), encoding="utf-8")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["figures"] = sum(chunk.modality is Modality.FIGURE for chunk in merged)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{doc_id}: pages={sorted(pages)} added={added} figures={report['figures']}")


if __name__ == "__main__":
    main()
