"""Audit completed real-corpus extraction and write machine-readable and Markdown reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from mmrag.pipeline import load_processed_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/corpus.csv"))
    parser.add_argument("--processed", type=Path, default=Path("data/real_processed"))
    parser.add_argument("--output", type=Path, default=Path("reports/real/extraction_review.json"))
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ready"]
    chunks = load_processed_chunks(args.processed)
    by_doc = {row["doc_id"]: [chunk for chunk in chunks if chunk.doc_id == row["doc_id"]] for row in rows}

    failures: list[str] = []
    documents: list[dict[str, object]] = []
    for row in rows:
        doc_id = row["doc_id"]
        report_path = args.processed / doc_id / "report.json"
        if not report_path.is_file():
            failures.append(f"{doc_id}: missing report.json")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        doc_chunks = by_doc[doc_id]
        counts = Counter(chunk.modality.value for chunk in doc_chunks)
        expected_pages = int(row["page_count"])
        if report.get("pages") != expected_pages:
            failures.append(f"{doc_id}: report pages={report.get('pages')} manifest pages={expected_pages}")
        if not counts["text"]:
            failures.append(f"{doc_id}: no text chunks")
        severe = [warning for warning in report.get("warnings", []) if "document skipped" in warning]
        if severe:
            failures.append(f"{doc_id}: {severe[0]}")
        documents.append(
            {
                "doc_id": doc_id,
                "company": row["company"],
                "pages": expected_pages,
                "chunks": sum(counts.values()),
                "text_chunks": counts["text"],
                "table_chunks": counts["table"],
                "figure_chunks": counts["figure"],
                "text_page_coverage": len({chunk.page for chunk in doc_chunks if chunk.modality.value == "text"}),
                "table_page_coverage": len({chunk.page for chunk in doc_chunks if chunk.modality.value == "table"}),
                "figure_page_coverage": len({chunk.page for chunk in doc_chunks if chunk.modality.value == "figure"}),
                "ocr_pages": report.get("ocr_pages", 0),
                "warning_count": len(report.get("warnings", [])),
                "warning_samples": report.get("warnings", [])[:10],
            }
        )

    payload = {
        "status": "pass" if not failures and len(documents) == len(rows) else "fail",
        "manifest_documents": len(rows),
        "reviewed_documents": len(documents),
        "total_pages": sum(int(item["pages"]) for item in documents),
        "total_chunks": sum(int(item["chunks"]) for item in documents),
        "failures": failures,
        "documents": documents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    lines = [
        "# Real-corpus extraction review",
        "",
        f"Status: **{payload['status']}**",
        "",
        "| Document | Pages | Text | Tables | Figures | OCR pages | Warnings |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in documents:
        lines.append(
            f"| {item['doc_id']} | {item['pages']} | {item['text_chunks']} | "
            f"{item['table_chunks']} | {item['figure_chunks']} | {item['ocr_pages']} | "
            f"{item['warning_count']} |"
        )
    if failures:
        lines.extend(("", "## Failures", "", *(f"- {failure}" for failure in failures)))
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"status={payload['status']} documents={len(documents)}/{len(rows)} "
        f"pages={payload['total_pages']} chunks={payload['total_chunks']} output={args.output}"
    )
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
