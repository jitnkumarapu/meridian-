"""Conservative finance-query normalization with visible expansions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    normalized: str
    expansions: tuple[str, ...]


_SYNONYMS = {
    "revenue": "sales",
    "sales": "revenue",
    "profit": "net income",
    "net income": "profit",
    "employees": "headcount",
    "headcount": "employees",
    "yoy": "year over year",
}


def normalize_query(query: str) -> NormalizedQuery:
    text = re.sub(r"\s+", " ", query).strip()
    text = re.sub(r"\b(?:fiscal\s+year|financial\s+year)\s*(20\d{2})\b", r"FY\1", text, flags=re.I)
    text = re.sub(r"\bFY\s*['\u2019]?(\d{2})\b", lambda m: f"FY20{m.group(1)}", text, flags=re.I)
    text = re.sub(r"\bper\s*cent\b", "%", text, flags=re.I)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*(million|billion|thousand|crore|lakh)s?\b", r"\1 \2", text, flags=re.I)
    lowered = text.casefold()
    expansions = tuple(value for key, value in _SYNONYMS.items() if re.search(rf"\b{re.escape(key)}\b", lowered))
    expanded = " ".join((text, *expansions)).strip()
    return NormalizedQuery(original=query, normalized=expanded, expansions=expansions)
