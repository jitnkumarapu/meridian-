from decimal import Decimal
from pathlib import Path

import pytest

from mmrag.finance import FinanceError, Operation, calculate, parse_financial_value
from mmrag.generation.reasoning import reason_financial_question
from mmrag.schemas import Chunk, Evidence, Modality


def _chunk(chunk_id: str = "c1", page: int = 2) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="acme-2024",
        page=page,
        modality=Modality.TABLE,
        content="evidence",
        source_path=Path("acme.pdf"),
    )


@pytest.mark.parametrize(
    ("display", "expected", "currency", "percentage"),
    [
        ("₹1,234.5 million", Decimal("1234500000.0"), "INR", False),
        ("(25) crore", Decimal("-250000000"), None, False),
        ("125 bps", Decimal("1.25"), None, True),
        ("12.4%", Decimal("12.4"), None, True),
        ("USD 2 billion", Decimal("2000000000"), "USD", False),
    ],
)
def test_financial_value_parser(display: str, expected: Decimal, currency: str | None, percentage: bool) -> None:
    value = parse_financial_value(display, _chunk())
    assert value.value == expected
    assert value.currency == currency
    assert value.percentage is percentage


def test_decimal_calculation_retains_evidence_trace() -> None:
    current = parse_financial_value("₹120 crore", _chunk("current"), fiscal_period="FY2024")
    prior = parse_financial_value("₹100 crore", _chunk("prior"), fiscal_period="FY2023")
    trace = calculate(Operation.PERCENTAGE_CHANGE, [current, prior])
    assert trace.result == Decimal("20.0")
    assert [item.source_chunk_id for item in trace.operands] == ["current", "prior"]


def test_financial_calculation_rejects_mixed_currency_and_zero_division() -> None:
    inr = parse_financial_value("INR 10", _chunk("inr"))
    usd = parse_financial_value("USD 2", _chunk("usd"))
    zero = parse_financial_value("INR 0", _chunk("zero"))
    with pytest.raises(FinanceError, match="incompatible"):
        calculate(Operation.SUM, [inr, usd])
    with pytest.raises(FinanceError, match="Division by zero"):
        calculate(Operation.RATIO, [inr, zero])


def test_retrieved_table_question_uses_deterministic_reasoning() -> None:
    chunk = _chunk()
    chunk.raw_grid = [["Metric", "FY2024", "FY2023"], ["Revenue", "120", "100"]]
    chunk.metadata = {"units": "INR crore"}
    result = reason_financial_question(
        "What is the percentage change in Revenue from FY2023 to FY2024?", [Evidence(chunk=chunk, fused_score=1)]
    )
    assert result is not None
    assert result.trace.result == Decimal("20.0")
    assert result.citations[0].chunk_id == chunk.chunk_id
    assert "[acme-2024, p.2, table]" in result.text


def test_reasoner_abstains_when_operation_or_row_is_not_explicit() -> None:
    chunk = _chunk()
    chunk.raw_grid = [["Metric", "FY2024", "FY2023"], ["Revenue", "120", "100"]]
    assert reason_financial_question("Tell me about revenue", [Evidence(chunk=chunk, fused_score=1)]) is None
    assert (
        reason_financial_question("What is the difference in profit?", [Evidence(chunk=chunk, fused_score=1)]) is None
    )
