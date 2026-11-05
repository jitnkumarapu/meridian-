from pathlib import Path

import pytest

from mmrag.ingestion import StaticOCRBackend
from mmrag.schemas import Modality


def test_static_ocr_backend() -> None:
    assert StaticOCRBackend("recognized", 91).extract(b"image", "eng") == ("recognized", 91)


def test_borderless_financial_table_signal() -> None:
    from mmrag.ingestion.pdf import _likely_borderless_table

    statement = "Year Ended 2024 2023 Revenue 245,122 211,915 Cost 74,114 65,863 Profit 88,136 72,361"
    assert _likely_borderless_table(statement)
    assert not _likely_borderless_table("Revenue increased during the year.")


def test_fixture_ingestion_and_cache(tmp_path: Path) -> None:
    pytest.importorskip("pymupdf")
    from mmrag.ingestion.pdf import ingest_pdf
    from scripts.make_fixtures import create_fixture

    fixture = create_fixture(tmp_path / "synthetic.pdf")
    chunks, report = ingest_pdf(
        fixture,
        tmp_path / "processed",
        ocr_backend=StaticOCRBackend("Employee retention improved to 94 percent in FY2024."),
    )
    assert report.pages == 3
    assert report.ocr_pages == 1
    assert {chunk.modality for chunk in chunks} >= {Modality.TEXT, Modality.TABLE, Modality.FIGURE}
    assert any("Revenue" in chunk.content for chunk in chunks if chunk.modality is Modality.TABLE)
    cached_chunks, cached_report = ingest_pdf(fixture, tmp_path / "processed")
    assert cached_report == report
    assert cached_chunks == chunks
