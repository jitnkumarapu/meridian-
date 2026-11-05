"""Versioned, deterministic grounded-answer prompts."""

from __future__ import annotations

import hashlib

from mmrag.schemas import Evidence, Modality

PROMPT_VERSION = "grounded-v3"


def _format_evidence(index: int, item: Evidence) -> str:
    chunk = item.chunk
    parts = [f"EVIDENCE {index} {chunk.citation_label}", f"CHUNK_ID: {chunk.chunk_id}"]
    if chunk.caption:
        parts.append(f"CAPTION: {chunk.caption}")
    if chunk.modality is Modality.TABLE and chunk.raw_grid:
        parts.append("GRID:\n" + "\n".join(" | ".join(row) for row in chunk.raw_grid))
    parts.append(f"CONTENT: {chunk.content}")
    return "\n".join(parts)


def build_prompt(query: str, evidence: list[Evidence], *, strict: bool = False, style: str = "compact") -> str:
    if style not in {"compact", "detailed"}:
        raise ValueError("answer style must be 'compact' or 'detailed'")
    rule = (
        "Your previous output failed citation validation. Use at least one citation copied exactly "
        "from an EVIDENCE header and do not cite anything else."
        if strict
        else "Cite every factual claim using a citation copied exactly from an EVIDENCE header."
    )
    blocks = "\n\n".join(_format_evidence(index, item) for index, item in enumerate(evidence, 1))
    return f"""PROMPT_VERSION: {PROMPT_VERSION}

## USER QUESTION
{query}

## RETRIEVED EVIDENCE
{blocks}

## GROUNDING RULES
Use only the retrieved evidence. Never use outside knowledge.
If evidence is insufficient, answer exactly: Insufficient evidence
Do not infer exact chart values unless they appear in caption, OCR, context, or a table.

## CITATION RULES
{rule}
Every citation must use this exact form: [document-id, p.12, text].
Copy document ID, page, and modality exactly from an EVIDENCE header.
Do not use nested brackets or evidence numbers.

## NUMERICAL RULES
Preserve currency, scale, unit, sign, and fiscal period.
Show the formula and evidence-backed operands for calculations. State any assumption.

## OUTPUT FORMAT
ANSWER_STYLE: {style}. {"Explain the evidence and arithmetic." if style == "detailed" else "Be concise."}
Write a calm, professional answer in Markdown: lead with the plain-language conclusion, use short
paragraphs, define unfamiliar finance terms on first use, and use a compact table when the evidence
contains comparable values. Use Markdown emphasis only for meaningful labels, never as decoration.
Do not expose internal labels such as "GROUNDING", "EVIDENCE", or "ANSWER", and do not dump raw
retrieved passages. End with one short, helpful next step when the evidence supports it.
Write the answer followed by canonical citations for every factual claim.

ANSWER:"""


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
