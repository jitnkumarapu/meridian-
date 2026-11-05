"""Conservative deterministic arithmetic over retrieved table cells."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mmrag.finance import FinanceError, FinancialValue, Operation, calculate, parse_financial_value
from mmrag.schemas import CalculationTrace, Citation, Evidence, Modality


@dataclass(frozen=True)
class DeterministicResult:
    text: str
    trace: CalculationTrace
    citations: list[Citation]


_OPERATIONS = (
    (re.compile(r"\bpercentage\s+(?:change|increase|decrease)|\bgrowth\s+rate\b", re.I), Operation.PERCENTAGE_CHANGE),
    (re.compile(r"\bdifference\b", re.I), Operation.DIFFERENCE),
    (re.compile(r"\bratio\b", re.I), Operation.RATIO),
    (re.compile(r"\b(?:sum|total)\s+of\b", re.I), Operation.SUM),
    (re.compile(r"\baverage\b", re.I), Operation.AVERAGE),
    (re.compile(r"\bminimum|min\b", re.I), Operation.MINIMUM),
    (re.compile(r"\bmaximum|max\b", re.I), Operation.MAXIMUM),
)


def _operation(query: str) -> Operation | None:
    return next((operation for pattern, operation in _OPERATIONS if pattern.search(query)), None)


def reason_financial_question(query: str, evidence: list[Evidence]) -> DeterministicResult | None:
    """Return a result only when operation, row, periods, values, and units are explicit."""
    operation = _operation(query)
    if operation is None:
        return None
    requested_periods = [period.upper().replace(" ", "") for period in re.findall(r"\bFY\s*\d{2,4}\b", query, re.I)]
    for item in evidence:
        chunk = item.chunk
        if chunk.modality is not Modality.TABLE or not chunk.raw_grid or len(chunk.raw_grid) < 2:
            continue
        headers = chunk.raw_grid[0]
        for row in chunk.raw_grid[1:]:
            row_label = row[0].strip() if row else ""
            if not row_label or row_label.casefold() not in query.casefold():
                continue
            values: list[FinancialValue] = []
            for column, display in enumerate(row[1:], 1):
                if column >= len(headers) or not display.strip():
                    continue
                header_period = headers[column].upper().replace(" ", "")
                if requested_periods and header_period not in requested_periods:
                    continue
                units = str(chunk.metadata.get("units", ""))
                scale = next(
                    (name for name in ("thousand", "lakh", "million", "crore", "billion") if name in units.casefold()),
                    None,
                )
                currency = next((code for code in ("INR", "USD", "EUR", "GBP") if code in units.upper()), None)
                try:
                    values.append(
                        parse_financial_value(
                            display,
                            chunk,
                            fiscal_period=headers[column],
                            row_label=row_label,
                            column_label=headers[column],
                            default_currency=currency,
                            default_scale=scale,
                        )
                    )
                except FinanceError:
                    continue
            required = 2 if operation in {Operation.DIFFERENCE, Operation.PERCENTAGE_CHANGE, Operation.RATIO} else 1
            if len(values) < required:
                continue
            try:
                trace = calculate(operation, values)
            except FinanceError:
                continue
            unit = trace.normalized_unit or ""
            rendered = format(trace.result, "f").rstrip("0").rstrip(".") or "0"
            citations = [
                Citation(
                    doc_id=value.doc_id,
                    page=value.page,
                    modality=Modality(value.modality),
                    chunk_id=value.source_chunk_id,
                )
                for value in values
            ]
            labels = " ".join(dict.fromkeys(citation.label for citation in citations))
            text = f"{row_label}: {rendered}{' ' + unit if unit else ''}. Calculation: {trace.formula}. {labels}"
            return DeterministicResult(text=text, trace=trace, citations=citations)
    return None
