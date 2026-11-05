"""Validate the manifest-named PDF corpus and optionally record immutable provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import fitz


def inspect_pdf(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with fitz.open(path) as document:
        if document.page_count < 1:
            raise ValueError("PDF contains no pages")
        return digest, document.page_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/corpus.csv"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--update", action="store_true", help="Write checksums, page counts, dates, and status")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "page_count" not in fieldnames:
        fieldnames.insert(fieldnames.index("sha256") + 1, "page_count")

    missing: list[str] = []
    invalid: list[str] = []
    ready = 0
    today = datetime.now(UTC).date().isoformat()
    for row in rows:
        filename = row["filename"]
        path = args.raw / filename
        if not path.is_file():
            missing.append(filename)
            if args.update:
                row["status"] = "missing"
            continue
        try:
            digest, pages = inspect_pdf(path)
        except (OSError, ValueError, fitz.FileDataError) as error:
            invalid.append(f"{filename}: {error}")
            if args.update:
                row["status"] = "invalid"
            continue
        ready += 1
        if args.update:
            row["retrieved_on"] = row.get("retrieved_on") or today
            row["sha256"] = digest
            row["page_count"] = str(pages)
            row["status"] = "ready"

    if args.update:
        with args.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"ready={ready} missing={len(missing)} invalid={len(invalid)} total={len(rows)}")
    for filename in missing:
        print(f"MISSING {filename}")
    for message in invalid:
        print(f"INVALID {message}")
    if missing or invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
