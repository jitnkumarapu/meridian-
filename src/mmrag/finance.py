"""Deterministic, evidence-bound financial value parsing and arithmetic."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from pydantic import BaseModel, Field

from mmrag.schemas import CalculationOperand, CalculationTrace, Chunk


class FinanceError(ValueError):
    """Raised when financial arithmetic would be ambiguous or unsafe."""


class Operation(StrEnum):
    DIFFERENCE = "difference"
    PERCENTAGE_CHANGE = "percentage_change"
    RATIO = "ratio"
    SUM = "sum"
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"


class FinancialValue(BaseModel):
    display_text: str
    value: Decimal
    currency: str | None = None
    scale: str | None = None
    unit: str | None = None
    percentage: bool = False
    fiscal_period: str | None = None
    row_label: str | None = None
    column_label: str | None = None
    source_chunk_id: str
    doc_id: str
    page: int = Field(ge=1)
    modality: str

    def operand(self) -> CalculationOperand:
        return CalculationOperand.model_validate(self.model_dump())


_SCALE = {
    "thousand": Decimal("1000"),
    "thousands": Decimal("1000"),
    "lakh": Decimal("100000"),
    "lakhs": Decimal("100000"),
    "million": Decimal("1000000"),
    "millions": Decimal("1000000"),
    "crore": Decimal("10000000"),
    "crores": Decimal("10000000"),
    "billion": Decimal("1000000000"),
    "billions": Decimal("1000000000"),
}
_VALUE = re.compile(
    r"(?P<currency>INR|USD|EUR|GBP|₹|\$|€|£)?\s*"
    r"(?P<number>\(?[-+]?\d[\d,]*(?:\.\d+)?\)?)\s*"
    r"(?P<scale>thousands?|lakhs?|millions?|crores?|billions?)?\s*"
    r"(?P<suffix>%|percent(?:age)?|bps?|basis\s+points?)?",
    re.IGNORECASE,
)
_CURRENCY = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP"}


def parse_financial_value(
    display_text: str,
    chunk: Chunk,
    *,
    fiscal_period: str | None = None,
    row_label: str | None = None,
    column_label: str | None = None,
    default_currency: str | None = None,
    default_scale: str | None = None,
    unit: str | None = None,
) -> FinancialValue:
    """Parse one explicit display value while retaining its evidence provenance."""
    match = _VALUE.search(display_text.strip())
    if match is None or not match.group("number"):
        raise FinanceError(f"No financial value found in {display_text!r}")
    number = match.group("number").replace(",", "")
    negative = number.startswith("(") and number.endswith(")")
    number = number.strip("()")
    try:
        numeric = Decimal(number)
    except InvalidOperation as exc:
        raise FinanceError(f"Invalid financial value {display_text!r}") from exc
    if negative:
        numeric = -numeric
    suffix = (match.group("suffix") or "").casefold()
    percentage = suffix.startswith("%") or suffix.startswith("percent")
    if suffix in {"bp", "bps", "basis point", "basis points"}:
        numeric /= Decimal("100")
        percentage = True
        unit = "percent"
    scale = (match.group("scale") or default_scale or "").casefold() or None
    if scale:
        numeric *= _SCALE[scale]
    raw_currency = match.group("currency") or default_currency
    currency = _CURRENCY.get(raw_currency or "", raw_currency.upper() if raw_currency else None)
    return FinancialValue(
        display_text=display_text,
        value=numeric,
        currency=currency,
        scale=scale,
        unit=unit or ("percent" if percentage else None),
        percentage=percentage,
        fiscal_period=fiscal_period,
        row_label=row_label,
        column_label=column_label,
        source_chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        page=chunk.page,
        modality=chunk.modality.value,
    )


def calculate(operation: Operation, values: list[FinancialValue]) -> CalculationTrace:
    """Calculate with Decimal only after unit, period, and provenance validation."""
    if not values:
        raise FinanceError("At least one evidence value is required")
    required = 2 if operation in {Operation.DIFFERENCE, Operation.PERCENTAGE_CHANGE, Operation.RATIO} else 1
    if len(values) < required:
        raise FinanceError(f"{operation.value} requires at least {required} operands")
    currencies = {item.currency for item in values if item.currency}
    units = {item.unit for item in values if item.unit}
    if len(currencies) > 1 or len(units) > 1:
        raise FinanceError("Operands have incompatible currencies or units")
    if operation in {Operation.SUM, Operation.AVERAGE, Operation.MINIMUM, Operation.MAXIMUM}:
        periods = {item.fiscal_period for item in values if item.fiscal_period}
        if len(periods) > 1:
            raise FinanceError("Aggregation operands have incompatible fiscal periods")
    numbers = [item.value for item in values]
    if operation is Operation.DIFFERENCE:
        result, formula = numbers[0] - numbers[1], "a - b"
    elif operation is Operation.PERCENTAGE_CHANGE:
        if numbers[1] == 0:
            raise FinanceError("Division by zero")
        result, formula = (numbers[0] - numbers[1]) / abs(numbers[1]) * Decimal("100"), "(a - b) / |b| x 100"
    elif operation is Operation.RATIO:
        if numbers[1] == 0:
            raise FinanceError("Division by zero")
        result, formula = numbers[0] / numbers[1], "a / b"
    elif operation is Operation.SUM:
        result, formula = sum(numbers, Decimal("0")), "sum(values)"
    elif operation is Operation.AVERAGE:
        result, formula = sum(numbers, Decimal("0")) / Decimal(len(numbers)), "sum(values) / count(values)"
    elif operation is Operation.MINIMUM:
        result, formula = min(numbers), "min(values)"
    else:
        result, formula = max(numbers), "max(values)"
    normalized_unit = "percent" if operation is Operation.PERCENTAGE_CHANGE else next(iter(units or currencies), None)
    return CalculationTrace(
        operation=operation.value,
        operands=[item.operand() for item in values],
        formula=formula,
        normalized_unit=normalized_unit,
        result=result,
        citations=[f"[{item.doc_id}, p.{item.page}, {item.modality}]" for item in values],
    )
