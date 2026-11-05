from pathlib import Path

from fastapi.testclient import TestClient

from mmrag.api import create_app
from mmrag.config import AppConfig
from mmrag.pipeline import MMRAGPipeline
from mmrag.schemas import Chunk, Modality


def _pipeline(tmp_path: Path) -> MMRAGPipeline:
    config = AppConfig.model_validate(
        {
            "paths": {
                "raw": tmp_path / "raw",
                "processed": tmp_path / "processed",
                "indexes": tmp_path / "indexes",
                "reports": tmp_path / "reports",
            },
            "ingestion": {},
            "embedding": {
                "text_model": "unused",
                "clip_model": "unused",
                "clip_pretrained": "unused",
                "test_mode": True,
            },
            "retrieval": {},
            "generation": {"provider": "mock", "model": "mock"},
            "baseline": {},
        }
    )
    chunks = [
        Chunk(
            chunk_id="acme-p1-text-1",
            doc_id="acme",
            page=1,
            modality=Modality.TEXT,
            content="Revenue was ₹1,234 million in FY2024.",
            source_path=tmp_path / "missing.pdf",
        )
    ]
    return MMRAGPipeline(config, chunks)


def test_health_retrieve_and_ask(tmp_path: Path) -> None:
    client = TestClient(create_app(_pipeline(tmp_path)))
    assert client.get("/health").json()["chunks"] == 1
    retrieved = client.post("/retrieve", json={"query": "FY2024 revenue", "k": 1})
    assert retrieved.status_code == 200
    assert retrieved.json()["evidence"][0]["chunk"]["chunk_id"] == "acme-p1-text-1"
    answer = client.post("/ask", json={"query": "FY2024 revenue", "k": 1})
    assert answer.status_code == 200
    assert answer.json()["grounded"] is True
    diagnostics = client.get("/diagnostics").json()
    assert diagnostics["active_model"] == "mock"
    assert "packages" in diagnostics


def test_explicit_company_name_routes_to_matching_document(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.chunks[0].doc_id = "acme-industries-2024"
    assert pipeline._infer_doc_ids("What revenue did Acme Industries report?") == {"acme-industries-2024"}
    assert pipeline._infer_doc_ids("What was total revenue?") is None


def test_missing_figure_and_page_are_404(tmp_path: Path) -> None:
    client = TestClient(create_app(_pipeline(tmp_path)))
    assert client.get("/figure/not-there").status_code == 404
    assert client.get("/page/not-there/1.png").status_code == 404


def test_page_endpoint_rejects_paths_outside_configured_roots(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.chunks[0].source_path = Path("C:/Windows/win.ini")
    client = TestClient(create_app(pipeline))
    assert client.get("/page/acme/1.png").status_code == 404


def test_figure_endpoint_rejects_paths_outside_processed_root(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.chunks.append(
        Chunk(
            chunk_id="acme-p1-figure-1",
            doc_id="acme",
            page=1,
            modality=Modality.FIGURE,
            content="",
            source_path=tmp_path / "missing.pdf",
            image_path=Path("C:/Windows/win.ini"),
        )
    )
    client = TestClient(create_app(pipeline))
    assert client.get("/figure/acme-p1-figure-1").status_code == 404


def test_upload_validation_and_job_lookup(tmp_path: Path) -> None:
    client = TestClient(create_app(_pipeline(tmp_path)))
    invalid = client.post("/jobs/upload?filename=report.pdf", content=b"not a pdf")
    assert invalid.status_code == 422
    wrong_extension = client.post("/jobs/upload?filename=report.html", content=b"%PDF-1.7")
    assert wrong_extension.status_code == 422
    assert client.get("/jobs/missing").status_code == 404


def test_upload_job_processes_and_deduplicates_pdf(tmp_path: Path) -> None:
    from scripts.make_fixtures import create_fixture

    source = create_fixture(tmp_path / "report.pdf")
    client = TestClient(create_app(_pipeline(tmp_path)))
    first = client.post("/jobs/upload?filename=report.pdf", content=source.read_bytes())
    assert first.status_code == 202
    job_id = first.json()["job_id"]
    assert first.json()["state"] == "queued"
    assert client.get(f"/jobs/{job_id}").json()["state"] == "completed"
    second = client.post("/jobs/upload?filename=duplicate.pdf", content=source.read_bytes())
    assert second.json()["job_id"] == job_id


def test_evaluation_endpoint_is_confined_to_reports(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    run = pipeline.config.paths.reports / "run-1"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"sample_size": 1}', encoding="utf-8")
    client = TestClient(create_app(pipeline))
    assert client.get("/evaluations").json()[0]["run_id"] == "run-1"
    assert client.get("/evaluations/../secrets").status_code == 404


def test_settings_history_and_streaming_contract(tmp_path: Path) -> None:
    client = TestClient(create_app(_pipeline(tmp_path)))
    settings = client.get("/settings").json()
    assert settings["demo_mode"] is True
    settings.update({"final_k": 2, "candidate_k": 4, "dense_weight": 0.5, "sparse_weight": 0.5})
    assert client.put("/settings", json=settings).json()["final_k"] == 2
    invalid = {**settings, "dense_weight": 0, "sparse_weight": 0}
    assert client.put("/settings", json=invalid).status_code == 422

    answer = client.post("/ask", json={"query": "revenue", "k": 1})
    assert answer.status_code == 200
    assert answer.json()["metadata"]["request_id"]
    history = client.get("/history").json()
    assert len(history) == 1
    history_id = history[0]["history_id"]
    updated = client.patch(f"/history/{history_id}", json={"title": "Pinned", "pinned": True})
    assert updated.json()["pinned"] is True
    assert client.post(f"/history/{history_id}/duplicate").status_code == 200
    assert client.delete(f"/history/{history_id}").status_code == 409
    assert client.delete(f"/history/{history_id}?confirm=true").status_code == 204
    assert client.delete("/history?confirm=true").status_code == 204

    streamed = client.post("/ask/stream", json={"query": "revenue", "k": 1})
    assert streamed.status_code == 200
    assert "event: provisional" in streamed.text
    assert "event: final" in streamed.text


def test_conversation_context_is_preserved_across_follow_up_questions(tmp_path: Path) -> None:
    client = TestClient(create_app(_pipeline(tmp_path)))
    first = client.post(
        "/ask",
        json={"query": "What was FY2024 revenue?", "k": 1, "conversation_id": "conversation-1"},
    )
    second = client.post(
        "/ask",
        json={"query": "And what source supports that?", "k": 1, "conversation_id": "conversation-1"},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["metadata"]["conversation_id"] == "conversation-1"
    assert second.json()["metadata"]["conversation_id"] == "conversation-1"


def test_document_detail_page_metadata_and_safe_removal(tmp_path: Path) -> None:
    from scripts.make_fixtures import create_fixture

    pipeline = _pipeline(tmp_path)
    raw = pipeline.config.paths.raw
    raw.mkdir(parents=True)
    fixture = create_fixture(raw / "acme.pdf")
    pipeline.chunks[0].source_path = fixture
    processed = pipeline.config.paths.processed / "acme"
    processed.mkdir(parents=True)
    (processed / "report.json").write_text(
        '{"doc_id":"acme","source_path":"' + str(fixture).replace("\\", "\\\\") + '","pages":3,"warnings":[]}',
        encoding="utf-8",
    )
    client = TestClient(create_app(pipeline))
    assert client.get("/documents/acme").status_code == 200
    metadata = client.get("/page/acme/1/metadata")
    assert metadata.status_code == 200
    assert metadata.json()["page_count"] == 3
    assert client.delete("/documents/acme").status_code == 409
    assert client.delete("/documents/acme?confirm=true").status_code == 204
    assert fixture.exists(), "removing an index entry must preserve the original PDF"
