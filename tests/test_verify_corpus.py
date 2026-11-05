import fitz

from scripts.verify_corpus import inspect_pdf


def test_inspect_pdf_returns_checksum_and_page_count(tmp_path) -> None:
    path = tmp_path / "report.pdf"
    document = fitz.open()
    document.new_page()
    document.save(path)
    document.close()

    digest, pages = inspect_pdf(path)

    assert len(digest) == 64
    assert pages == 1
