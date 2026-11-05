from pathlib import Path

import pytest

from mmrag.baseline import to_text_only
from mmrag.generation import AnswerPipeline, MockLLM, parse_citations, validate_citations
from mmrag.generation.llm import GeminiClient, LLMError, create_llm_client, provider_health
from mmrag.schemas import Chunk, Evidence, Modality


def chunk(modality: Modality = Modality.TABLE) -> Chunk:
    return Chunk(
        chunk_id="acme-p2-table-1",
        doc_id="acme-2024",
        page=2,
        modality=modality,
        content="Revenue | FY2024 = ₹120 crore; FY2023 = ₹100 crore",
        raw_grid=[["Metric", "FY2024", "FY2023"], ["Revenue", "120", "100"]] if modality is Modality.TABLE else None,
        source_path=Path("acme.pdf"),
    )


def evidence() -> list[Evidence]:
    return [Evidence(chunk=chunk(), fused_score=1.0)]


def test_parse_and_resolve_citations() -> None:
    text = "Revenue increased [acme-2024, p.2, table]."
    assert parse_citations(text)[0].page == 2
    result = validate_citations(text, evidence())
    assert result.valid
    assert result.citations[0].chunk_id == "acme-p2-table-1"


def test_nested_evidence_label_resolves_citation() -> None:
    result = validate_citations("Revenue [EVIDENCE 1 [acme-2024, p.2, table]].", evidence())
    assert result.valid
    assert result.citations[0].chunk_id == "acme-p2-table-1"


def test_invalid_citation_retries_then_rejects() -> None:
    llm = MockLLM(["Wrong [other, p.2, table]", "Still wrong [acme-2024, p.9, table]"])
    answer = AnswerPipeline(llm).answer("What was revenue?", evidence())
    assert answer.text == "Insufficient evidence"
    assert not answer.grounded
    assert llm.calls == 2


def test_retry_can_recover() -> None:
    llm = MockLLM(["No citation", "₹120 crore [acme-2024, p.2, table]"])
    answer = AnswerPipeline(llm).answer("What was revenue?", evidence())
    assert answer.grounded
    assert answer.citations[0].chunk_id == "acme-p2-table-1"
    assert llm.calls == 2


def test_baseline_drops_figures_and_flattens_tables() -> None:
    transformed = to_text_only([chunk(), chunk(Modality.FIGURE)])
    assert len(transformed) == 1
    assert transformed[0].modality is Modality.TEXT
    assert transformed[0].raw_grid is None
    assert transformed[0].content == "Metric FY2024 FY2023 Revenue 120 100"


def test_missing_provider_key_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        create_llm_client("anthropic", "test-model")


def test_missing_gemini_key_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        create_llm_client("gemini", "gemini-2.5-flash")


def test_gemini_client_supports_text_and_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    captured: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "candidates": [{"content": {"parts": [{"text": "Grounded answer"}]}}],
                "usageMetadata": {"totalTokenCount": 4},
            }

    def post(*args: object, **kwargs: object) -> Response:
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("mmrag.generation.llm.httpx.post", post)
    client = GeminiClient("gemini-2.5-flash", api_key="test-key")
    result = client.complete("Question", image_paths=[str(image)])
    assert result.text == "Grounded answer"
    assert captured["headers"] == {"x-goog-api-key": "test-key"}
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert len(payload["contents"][0]["parts"]) == 2


def test_provider_health_never_serializes_key(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "super-secret-provider-key"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    health = provider_health("gemini", use_vision=True)
    assert health["status"] == "ready"
    assert secret not in str(health)


def test_answer_log_contains_metadata_but_no_provider_key(caplog: pytest.LogCaptureFixture) -> None:
    secret = "never-log-this-key"
    with caplog.at_level("INFO"):
        answer = AnswerPipeline(
            MockLLM(["Revenue [acme-2024, p.2, table]"], model_id="safe-model"), provider="mock"
        ).answer("Revenue?", evidence())
    assert answer.grounded
    assert '"model": "safe-model"' in caplog.text
    assert '"grounded": true' in caplog.text
    assert secret not in caplog.text
