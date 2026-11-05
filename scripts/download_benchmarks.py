"""Download explicitly licensed benchmark files and record immutable provenance.

The URLs are intentionally configurable because benchmark hosting and click-through terms change.
This script never bypasses authentication or accepts terms on a user's behalf.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def download(name: str, url: str, output: Path) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "mmrag-finance-research/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    output.write_bytes(payload)
    return {
        "name": name,
        "url": url,
        "retrieved_on": datetime.now(UTC).date().isoformat(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "path": output.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=["finqa", "tatqa", "docvqa", "slidevqa"])
    parser.add_argument("url", help="Official or Hugging Face resolved file URL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entry = download(args.name, args.url, args.output)
    manifest = args.output.parent / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
