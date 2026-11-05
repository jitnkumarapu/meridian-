"""Deterministic, sentence-aware text chunking."""

from __future__ import annotations

import re
from collections.abc import Iterable

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹$€£(])")
_HEADING = re.compile(r"^[A-Z][A-Z0-9 &/()\-]{2,80}$")


def _sentences(text: str) -> list[str]:
    clean = re.sub(r"[ \t]+", " ", text.replace("\u00ad", ""))
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    result: list[str] = []
    for paragraph in clean.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        result.extend(part.strip() for part in _SENTENCE_END.split(paragraph) if part.strip())
    return result


def chunk_text(text: str, max_tokens: int = 300, overlap_tokens: int = 45) -> list[str]:
    """Split text without cutting sentences unless one sentence exceeds the budget.

    Token counts use whitespace tokens deliberately: the extraction stage must stay independent of a
    particular model tokenizer, while the configured overlap remains deterministic and predictable.
    """
    if max_tokens < 1 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError("expected max_tokens > 0 and 0 <= overlap_tokens < max_tokens")
    sentences = _sentences(text)
    if not sentences:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_count = 0
    for sentence in sentences:
        words = sentence.split()
        if len(words) > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_count = [], 0
            step = max_tokens - overlap_tokens
            chunks.extend(" ".join(words[i : i + max_tokens]) for i in range(0, len(words), step))
            continue
        if current and current_count + len(words) > max_tokens:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            count = 0
            for old_sentence in reversed(current):
                size = len(old_sentence.split())
                if count + size > overlap_tokens:
                    break
                overlap.insert(0, old_sentence)
                count += size
            current, current_count = overlap, count
        current.append(sentence)
        current_count += len(words)
    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def nearest_heading(blocks: Iterable[str]) -> str:
    """Return the last heading-like block in reading order."""
    heading = ""
    for block in blocks:
        candidate = block.strip()
        if _HEADING.fullmatch(candidate):
            heading = candidate
    return heading
