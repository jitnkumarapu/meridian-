"""Download manifest-declared public PDFs with bounded streaming and PDF validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import fitz
import httpx

MAX_PDF_BYTES = 200 * 1024 * 1024


def _inspect(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError("response does not start with a PDF signature")
        handle.seek(0)
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    with fitz.open(path) as document:
        if document.page_count < 1:
            raise ValueError("PDF contains no pages")
        return digest.hexdigest(), document.page_count


def _download(client: httpx.Client, url: str, target: Path) -> tuple[str, int]:
    temporary = target.with_suffix(target.suffix + ".part")
    written = 0
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for block in response.iter_bytes(1024 * 1024):
                    written += len(block)
                    if written > MAX_PDF_BYTES:
                        raise ValueError("PDF exceeds the 200 MiB corpus limit")
                    handle.write(block)
        digest, pages = _inspect(temporary)
        temporary.replace(target)
        return digest, pages
    finally:
        temporary.unlink(missing_ok=True)


def download(manifest: Path, output_dir: Path, timeout: float = 90) -> tuple[int, int]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError("Corpus manifest is empty")
    if "page_count" not in fieldnames:
        fieldnames.insert(fieldnames.index("sha256") + 1, "page_count")

    completed = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "mmrag-finance-research/0.1 (public annual-report corpus)"}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout) as client:
        for row in rows:
            target = output_dir / row["filename"]
            try:
                if target.is_file():
                    digest, pages = _inspect(target)
                else:
                    print(f"DOWNLOADING {row['filename']}")
                    digest, pages = _download(client, row["url"], target)
                row["sha256"] = digest
                row["page_count"] = str(pages)
                row["retrieved_on"] = row.get("retrieved_on") or datetime.now(UTC).date().isoformat()
                row["status"] = "ready"
                completed += 1
                print(f"READY {row['filename']} pages={pages}")
            except (OSError, ValueError, fitz.FileDataError, httpx.HTTPError) as error:
                row["status"] = "manual"
                row["notes"] = f"{row.get('notes', '')}; download failed: {error}".strip("; ")
                print(f"MANUAL {row['filename']}: {error}")

    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return completed, len(rows) - completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/corpus.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    completed, manual = download(args.manifest, args.output, args.timeout)
    print(f"ready={completed} manual={manual} total={completed + manual}")
    if manual:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
