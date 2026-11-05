"""Parse and resolve the project's canonical citation format."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mmrag.schemas import Citation, Evidence, Modality

_CITATION_PATTERN = re.compile(
    r"\[(?P<doc_id>[^\][,\n]+),\s*p\.(?P<page>\d+),\s*"
    r"(?P<modality>text|table|figure)\]",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationValidation:
    valid: bool
    citations: list[Citation]
    invalid_labels: list[str]


def parse_citations(text: str) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[tuple[str, int, Modality]] = set()
    for match in _CITATION_PATTERN.finditer(text):
        citation = Citation(
            doc_id=match.group("doc_id").strip(),
            page=int(match.group("page")),
            modality=Modality(match.group("modality").lower()),
        )
        key = (citation.doc_id, citation.page, citation.modality)
        if key not in seen:
            seen.add(key)
            citations.append(citation)
    return citations


def validate_citations(text: str, evidence: list[Evidence]) -> CitationValidation:
    parsed = parse_citations(text)
    lookup: dict[tuple[str, int, Modality], list[str]] = {}
    for item in evidence:
        key = (item.chunk.doc_id, item.chunk.page, item.chunk.modality)
        lookup.setdefault(key, []).append(item.chunk.chunk_id)

    resolved: list[Citation] = []
    invalid: list[str] = []
    for citation in parsed:
        key = (citation.doc_id, citation.page, citation.modality)
        chunk_ids = lookup.get(key)
        if not chunk_ids:
            invalid.append(citation.label)
        else:
            resolved.append(citation.model_copy(update={"chunk_id": chunk_ids[0]}))
    return CitationValidation(valid=bool(resolved) and not invalid, citations=resolved, invalid_labels=invalid)
