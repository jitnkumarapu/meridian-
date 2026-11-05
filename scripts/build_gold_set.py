"""Validate and deterministically balance human-authored own-corpus gold questions."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

TYPES = {"text", "table", "figure", "multimodal"}


def validate(records: list[dict[str, object]], *, require_verified: bool = True) -> None:
    errors: list[str] = []
    for index, row in enumerate(records, 1):
        missing = {
            "id",
            "question",
            "reference_answer",
            "gold_document",
            "gold_pages",
            "gold_modalities",
            "gold_chunk_ids",
            "type",
            "author",
            "reviewer",
            "verified",
        } - row.keys()
        if missing:
            errors.append(f"row {index}: missing {sorted(missing)}")
        if row.get("type") not in TYPES:
            errors.append(f"row {index}: invalid type {row.get('type')!r}")
        if row.get("verified") is True and not row.get("reviewer"):
            errors.append(f"row {index}: verified without reviewer")
        if require_verified and row.get("verified") is not True:
            errors.append(f"row {index}: final gold set requires verified=true")
    if errors:
        raise ValueError("\n".join(errors))


def normalize(row: dict[str, object]) -> dict[str, object]:
    """Convert the human-review schema into the evaluator's stable input schema."""
    return {
        "question_id": row["id"],
        "question": row["question"],
        "relevant_chunk_ids": row["gold_chunk_ids"],
        "reference_answer": row["reference_answer"],
        "question_type": row["type"],
        "verified": row["verified"],
        "reference_document": row["gold_document"],
        "reference_pages": row["gold_pages"],
        "reference_modalities": row["gold_modalities"],
        "units": row.get("units"),
        "fiscal_period": row.get("fiscal_period"),
        "author": row["author"],
        "reviewer": row["reviewer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL authored by the team")
    parser.add_argument("--output", type=Path, default=Path("data/eval/own_corpus_gold.jsonl"))
    parser.add_argument("--dev-output", type=Path, default=Path("data/eval/own_corpus_dev.jsonl"))
    parser.add_argument("--test-output", type=Path, default=Path("data/eval/own_corpus_locked_test.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Build clearly labelled draft splits without treating them as final human-verified gold data",
    )
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate(records, require_verified=not args.allow_unverified)
    counts = Counter(str(row["type"]) for row in records)
    if len(records) < 60 or any(counts[kind] < 12 for kind in TYPES):
        raise ValueError(f"need >=60 questions and >=12/type; got total={len(records)}, counts={dict(counts)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized = [normalize(row) for row in records]
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized), encoding="utf-8")
    rng = random.Random(args.seed)
    dev: list[dict[str, object]] = []
    test: list[dict[str, object]] = []
    for kind in sorted(TYPES):
        subset = [row for row in normalized if row["question_type"] == kind]
        rng.shuffle(subset)
        cutoff = max(1, round(len(subset) * 0.7))
        dev.extend(subset[:cutoff])
        test.extend(subset[cutoff:])
    args.dev_output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in dev), encoding="utf-8")
    args.test_output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in test), encoding="utf-8")


if __name__ == "__main__":
    main()
