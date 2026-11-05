from mmrag.extraction.tables import normalize_grid, score_table, serialize_table
from mmrag.extraction.text import chunk_text


def test_table_serialization_preserves_financial_values_and_labels() -> None:
    grid = [
        ["Metric", "FY2023", "FY2024"],
        ["Revenue", "₹1,100.5", "₹1,234.5"],
        ["Loss", "(120)", "(40)"],
    ]
    result = serialize_table(grid, "Results", "₹ millions", 7)
    assert "Revenue | FY2023 = ₹1,100.5; FY2024 = ₹1,234.5" in result
    assert "Loss | FY2023 = (120); FY2024 = (40)" in result
    assert "Units: ₹ millions" in result


def test_normalize_grid_propagates_merged_header() -> None:
    assert normalize_grid([["Metric", "FY2024", None], ["Revenue", "10", "20"]])[0] == [
        "Metric",
        "FY2024",
        "FY2024",
    ]


def test_chunk_text_is_sentence_aware_and_overlaps() -> None:
    text = "Alpha grew strongly. Beta remained stable. Gamma improved materially. Delta was unchanged."
    chunks = chunk_text(text, max_tokens=7, overlap_tokens=3)
    assert len(chunks) >= 2
    assert chunks[0].endswith(".")
    assert all(len(chunk.split()) <= 7 for chunk in chunks)


def test_chunk_text_validates_budget() -> None:
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello", max_tokens=5, overlap_tokens=5)


def test_table_quality_warns_about_sparse_extraction() -> None:
    quality = score_table([["Metric", "FY24", "FY23"], ["Revenue", "", ""], ["", "", ""]])
    assert 0 < quality.score < 1
    assert "many blank cells" in quality.warnings
