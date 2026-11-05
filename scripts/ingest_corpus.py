"""Ingest exactly the ready manifest corpus, in parallel, with per-document caching."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mmrag.config import load_config
from mmrag.ingestion import TesseractOCRBackend, ingest_pdf


def _ingest_one(config_path: str, source: str, force: bool) -> dict[str, object]:
    config = load_config(Path(config_path))
    try:
        ocr_backend = TesseractOCRBackend()
    except RuntimeError:
        ocr_backend = None
    _chunks, report = ingest_pdf(
        Path(source),
        config.paths.processed,
        min_text_chars=config.ingestion.min_text_chars,
        chunk_tokens=config.ingestion.chunk_tokens,
        overlap_tokens=config.ingestion.overlap_tokens,
        ocr_language=config.ingestion.ocr_language,
        ocr_backend=ocr_backend,
        force=force,
        max_pages=config.ingestion.max_pdf_pages,
    )
    return report.model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/corpus.csv"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/production.yaml"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--documents", nargs="*", help="Optional manifest document IDs to ingest")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ready"]
    sources: list[Path] = []
    for row in rows:
        if args.documents and row.get("doc_id") not in args.documents:
            continue
        filename = row.get("filename", "").strip()
        if not filename or Path(filename).name != filename:
            parser.error(f"unsafe manifest filename: {filename!r}")
        source = args.raw / filename
        if not source.is_file():
            parser.error(f"manifest file is missing: {source}")
        sources.append(source)

    failures = 0
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(sources))) as executor:
        future_paths = {
            executor.submit(_ingest_one, str(args.config.resolve()), str(path.resolve()), args.force): path
            for path in sources
        }
        for future in as_completed(future_paths):
            path = future_paths[future]
            try:
                report = future.result()
            except Exception as error:
                failures += 1
                print(f"FAILED {path.name}: {type(error).__name__}: {error}", flush=True)
                continue
            results.append(report)
            print(
                "READY "
                f"{path.name} pages={report['pages']} text={report['text_chunks']} "
                f"tables={report['tables']} figures={report['figures']} "
                f"warnings={len(report['warnings'])}",
                flush=True,
            )

    print(json.dumps(sorted(results, key=lambda item: str(item["doc_id"])), indent=2))
    print(f"completed={len(results)} failed={failures} total={len(sources)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
