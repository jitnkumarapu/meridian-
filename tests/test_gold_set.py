import pytest

from scripts.build_gold_set import normalize, validate


def test_human_gold_record_normalizes_for_evaluator() -> None:
    record = {
        "id": "infosys-2024-table-01",
        "question": "What was consolidated revenue in FY2024?",
        "reference_answer": "INR 153,670 crore",
        "gold_document": "infosys-2024",
        "gold_pages": [142],
        "gold_modalities": ["table"],
        "gold_chunk_ids": ["infosys-2024-p142-table-1"],
        "type": "table",
        "author": "Author A",
        "reviewer": "Reviewer B",
        "verified": True,
        "units": "INR crore",
        "fiscal_period": "FY2024",
    }

    validate([record])
    normalized = normalize(record)

    assert normalized["question_id"] == "infosys-2024-table-01"
    assert normalized["relevant_chunk_ids"] == ["infosys-2024-p142-table-1"]
    assert normalized["question_type"] == "table"
    assert normalized["reference_document"] == "infosys-2024"


def test_unverified_draft_requires_explicit_opt_in() -> None:
    record = {
        "id": "draft-1",
        "question": "Draft?",
        "reference_answer": "Draft.",
        "gold_document": "doc",
        "gold_pages": [1],
        "gold_modalities": ["text"],
        "gold_chunk_ids": ["doc-p1-text-1"],
        "type": "text",
        "author": "AI draft",
        "reviewer": "PENDING INDEPENDENT HUMAN REVIEW",
        "verified": False,
    }

    with pytest.raises(ValueError, match="verified=true"):
        validate([record])
    validate([record], require_verified=False)
