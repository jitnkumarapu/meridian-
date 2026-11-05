"""Typed HTTP API shared by the React and Streamlit clients."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, cast

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mmrag.generation import provider_health
from mmrag.generation.llm import LLMError
from mmrag.history import HistoryEntry, HistoryStore
from mmrag.pipeline import MMRAGPipeline
from mmrag.schemas import Answer, Modality, RetrievalResult


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    baseline: bool = False
    modalities: set[Modality] | None = None
    doc_ids: set[str] | None = None
    fiscal_years: set[str] | None = None
    style: str = Field(default="compact", pattern="^(compact|detailed)$")
    request_id: str | None = Field(default=None, max_length=80)
    conversation_id: str | None = Field(default=None, max_length=80)


class JobRecord(BaseModel):
    job_id: str
    filename: str
    checksum: str
    state: str
    progress: int = Field(ge=0, le=100)
    created_at: str
    updated_at: str
    error: str | None = None
    cancel_requested: bool = False
    stage: str = "queued"
    pages_processed: int = 0
    total_pages: int | None = None
    warning_count: int = 0


class RuntimeSettings(BaseModel):
    final_k: int = Field(ge=1, le=50)
    candidate_k: int = Field(ge=1, le=200)
    fusion: Literal["rrf", "weighted"]
    rrf_k: int = Field(ge=1, le=1000)
    dense_weight: float = Field(ge=0, le=1)
    sparse_weight: float = Field(ge=0, le=1)
    provider: str
    model: str
    vision: bool
    demo_mode: bool


class HistoryUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    pinned: bool | None = None


def create_app(pipeline: MMRAGPipeline | None = None) -> FastAPI:
    application = FastAPI(title="MMRAG Finance API", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )
    state: dict[str, MMRAGPipeline] = {}
    jobs: dict[str, JobRecord] = {}
    job_sources: dict[str, Path] = {}
    jobs_lock = threading.Lock()
    history_store: HistoryStore | None = None
    conversations: dict[str, list[tuple[str, str]]] = {}
    if pipeline is not None:
        state["pipeline"] = pipeline

    def get_pipeline() -> MMRAGPipeline:
        if "pipeline" not in state:
            config_path = Path(os.getenv("MMRAG_CONFIG", "configs/default.yaml"))
            state["pipeline"] = MMRAGPipeline.from_config(config_path)
        return state["pipeline"]

    def get_history() -> HistoryStore:
        nonlocal history_store
        if history_store is None:
            history_store = HistoryStore(get_pipeline().config.paths.processed / "app_history.sqlite3")
        return history_store

    def conversation_query(request: SearchRequest) -> tuple[str, str]:
        conversation_id = request.conversation_id or uuid.uuid4().hex
        turns = conversations.setdefault(conversation_id, [])
        if not turns:
            return request.query, conversation_id
        context = "\n".join(
            f"Earlier question: {question}\nEarlier answer: {answer}" for question, answer in turns[-3:]
        )
        return f"Conversation context:\n{context}\n\nCurrent question: {request.query}", conversation_id

    def approved_path(path: Path, *roots: Path) -> Path | None:
        resolved = path.resolve()
        for root in roots:
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                continue
            return resolved
        return None

    def update_job(job_id: str, **changes: object) -> JobRecord:
        with jobs_lock:
            current = jobs[job_id]
            updated = current.model_copy(update={**changes, "updated_at": datetime.now(UTC).isoformat()})
            jobs[job_id] = updated
            return updated

    def process_job(job_id: str) -> None:
        record = update_job(job_id, state="processing", stage="validating", progress=5)
        if record.cancel_requested:
            update_job(job_id, state="cancelled", progress=record.progress)
            return
        try:
            update_job(job_id, stage="extracting", progress=15)
            reports = get_pipeline().ingest(job_sources[job_id])
            total_pages = sum(report.pages for report in reports)
            warnings = sum(len(report.warnings) for report in reports)
            record = update_job(
                job_id,
                stage="building_embeddings",
                progress=75,
                pages_processed=total_pages,
                total_pages=total_pages,
                warning_count=warnings,
            )
            if record.cancel_requested:
                update_job(job_id, state="cancelled")
                return
            get_pipeline().build_indexes()
            update_job(job_id, state="completed", stage="complete", progress=100)
        except (RuntimeError, ValueError, OSError) as error:
            update_job(job_id, state="failed", stage="failed", error=f"{type(error).__name__}: {error}")

    @application.get("/health")
    def health() -> dict[str, object]:
        service = get_pipeline()
        return {
            "status": "ok",
            "chunks": len(service.chunks),
            "demo_mode": service.config.generation.provider == "mock",
            "provider": provider_health(
                service.config.generation.provider,
                use_vision=service.config.generation.use_vision,
                key_env=service.config.generation.key_env,
            ),
        }

    @application.get("/diagnostics")
    def diagnostics() -> dict[str, object]:
        service = get_pipeline()
        packages: dict[str, str] = {}
        for package in ("mmrag-finance", "pymupdf", "pdfplumber", "sentence-transformers", "open-clip-torch"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "not-installed"
        index_path = service.config.paths.indexes / "multimodal" / "index_metadata.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else None
        return {
            "provider": provider_health(
                service.config.generation.provider,
                use_vision=service.config.generation.use_vision,
                key_env=service.config.generation.key_env,
            ),
            "active_model": service.config.generation.model,
            "text_encoder": service.config.embedding.text_model,
            "figure_encoder": f"{service.config.embedding.clip_model}:{service.config.embedding.clip_pretrained}",
            "index": index,
            "packages": packages,
        }

    @application.get("/documents")
    def documents() -> list[dict[str, object]]:
        return get_pipeline().documents()

    @application.get("/documents/{doc_id}")
    def document_detail(doc_id: str) -> dict[str, object]:
        matching = [chunk for chunk in get_pipeline().chunks if chunk.doc_id == doc_id]
        report = next((item for item in get_pipeline().documents() if item.get("doc_id") == doc_id), None)
        if not matching or report is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "document": report,
            "tables": [chunk.model_dump(mode="json") for chunk in matching if chunk.modality is Modality.TABLE],
            "figures": [chunk.model_dump(mode="json") for chunk in matching if chunk.modality is Modality.FIGURE],
        }

    @application.post("/documents/{doc_id}/reindex")
    def reindex_document(doc_id: str) -> dict[str, object]:
        if not any(chunk.doc_id == doc_id for chunk in get_pipeline().chunks):
            raise HTTPException(status_code=404, detail="Document not found")
        target = get_pipeline().build_indexes()
        return {"status": "complete", "index": str(target), "document": doc_id}

    @application.post("/documents/{doc_id}/reingest", response_model=JobRecord, status_code=202)
    def reingest_document(doc_id: str, background_tasks: BackgroundTasks) -> JobRecord:
        chunk = next((item for item in get_pipeline().chunks if item.doc_id == doc_id), None)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Document not found")
        source = approved_path(chunk.source_path, get_pipeline().config.paths.raw)
        if source is None or not source.is_file():
            raise HTTPException(status_code=409, detail="Original PDF is unavailable")
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        now = datetime.now(UTC).isoformat()
        job = JobRecord(
            job_id=uuid.uuid4().hex,
            filename=source.name,
            checksum=checksum,
            state="queued",
            progress=0,
            created_at=now,
            updated_at=now,
        )
        with jobs_lock:
            jobs[job.job_id] = job
            job_sources[job.job_id] = source
        background_tasks.add_task(process_job, job.job_id)
        return job

    @application.delete("/documents/{doc_id}", status_code=204)
    def remove_document(doc_id: str, confirm: bool = False) -> Response:
        if not confirm:
            raise HTTPException(status_code=409, detail="Removal requires confirm=true")
        service = get_pipeline()
        if not any(chunk.doc_id == doc_id for chunk in service.chunks):
            raise HTTPException(status_code=404, detail="Document not found")
        processed_dir = approved_path(service.config.paths.processed / doc_id, service.config.paths.processed)
        if processed_dir is not None and processed_dir.is_dir():
            shutil.rmtree(processed_dir)
        service.remove_document_from_index(doc_id)
        return Response(status_code=204)

    @application.post("/retrieve", response_model=RetrievalResult)
    def retrieve(request: SearchRequest) -> RetrievalResult:
        try:
            return get_pipeline().retrieve(
                request.query,
                k=request.k,
                baseline=request.baseline,
                modalities=request.modalities,
                doc_ids=request.doc_ids,
                fiscal_years=request.fiscal_years,
            )
        except (RuntimeError, ValueError, LLMError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.post("/ask", response_model=Answer)
    def ask(request: SearchRequest) -> Answer:
        try:
            query, conversation_id = conversation_query(request)
            answer = get_pipeline().ask(
                query,
                k=request.k,
                baseline=request.baseline,
                modalities=request.modalities,
                doc_ids=request.doc_ids,
                fiscal_years=request.fiscal_years,
                style=request.style,
            )
            request_id = request.request_id or uuid.uuid4().hex
            answer.metadata["request_id"] = request_id
            answer.metadata["conversation_id"] = conversation_id
            conversations[conversation_id].append((request.query, answer.text))
            get_history().add(
                request.query,
                {
                    "answer": answer.text,
                    "citations": [citation.model_dump(mode="json") for citation in answer.citations],
                    "evidence_ids": [item.chunk.chunk_id for item in answer.evidence],
                    "settings": request.model_dump(mode="json", exclude={"request_id"}),
                    "document_scope": sorted(request.doc_ids or set()),
                    "latency_ms": answer.latency_ms,
                    "grounded": answer.grounded,
                    "provider": answer.provider,
                    "model": answer.model_id,
                },
            )
            return answer
        except (RuntimeError, ValueError, LLMError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.post("/ask/stream")
    def ask_stream(request: SearchRequest) -> StreamingResponse:
        """SSE contract: provisional text is never marked grounded; final answer passed the citation gate."""
        try:
            query, conversation_id = conversation_query(request)
            answer = get_pipeline().ask(
                query,
                k=request.k,
                baseline=request.baseline,
                modalities=request.modalities,
                doc_ids=request.doc_ids,
                fiscal_years=request.fiscal_years,
                style=request.style,
            )
            answer.metadata["conversation_id"] = conversation_id
            conversations[conversation_id].append((request.query, answer.text))
        except (RuntimeError, ValueError, LLMError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        def events() -> Iterator[str]:
            words = answer.text.split()
            for offset in range(0, len(words), 12):
                yield (
                    "event: provisional\ndata: "
                    + json.dumps({"text": " ".join(words[: offset + 12]), "grounded": False})
                    + "\n\n"
                )
            yield "event: final\ndata: " + answer.model_dump_json() + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @application.post("/jobs/upload", response_model=JobRecord, status_code=202)
    async def upload_job(request: Request, background_tasks: BackgroundTasks, filename: str) -> JobRecord:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).name).strip(".-")
        if not safe_name.lower().endswith(".pdf") or not safe_name:
            raise HTTPException(status_code=422, detail="Only a named PDF file is accepted")
        payload = await request.body()
        max_mib = get_pipeline().config.ingestion.max_upload_mib
        if len(payload) > max_mib * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"PDF exceeds the {max_mib} MiB upload limit")
        if not payload.startswith(b"%PDF-"):
            raise HTTPException(status_code=422, detail="Upload is not a valid PDF payload")
        try:
            import pymupdf

            with pymupdf.open(stream=payload, filetype="pdf") as uploaded:  # type: ignore[no-untyped-call]
                if uploaded.needs_pass or uploaded.page_count < 1:
                    raise HTTPException(status_code=422, detail="Encrypted or empty PDFs are not accepted")
                if uploaded.page_count > get_pipeline().config.ingestion.max_pdf_pages:
                    raise HTTPException(status_code=422, detail="PDF page count exceeds the configured limit")
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=422, detail="Upload is not a readable PDF") from error
        checksum = hashlib.sha256(payload).hexdigest()
        existing = next((job for job in jobs.values() if job.checksum == checksum), None)
        if existing and existing.state in {"queued", "processing", "completed"}:
            return existing
        service = get_pipeline()
        service.config.paths.raw.mkdir(parents=True, exist_ok=True)
        target = service.config.paths.raw / safe_name
        if approved_path(target, service.config.paths.raw) is None:
            raise HTTPException(status_code=422, detail="Invalid upload filename")
        target.write_bytes(payload)
        now = datetime.now(UTC).isoformat()
        job = JobRecord(
            job_id=uuid.uuid4().hex,
            filename=safe_name,
            checksum=checksum,
            state="queued",
            progress=0,
            created_at=now,
            updated_at=now,
        )
        with jobs_lock:
            jobs[job.job_id] = job
            job_sources[job.job_id] = target
        background_tasks.add_task(process_job, job.job_id)
        return job

    @application.get("/jobs", response_model=list[JobRecord])
    def list_jobs() -> list[JobRecord]:
        return sorted(jobs.values(), key=lambda job: job.created_at, reverse=True)

    @application.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return jobs[job_id]

    @application.post("/jobs/{job_id}/cancel", response_model=JobRecord)
    def cancel_job(job_id: str) -> JobRecord:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        current = jobs[job_id]
        if current.state in {"completed", "failed", "cancelled"}:
            raise HTTPException(status_code=409, detail=f"Cannot cancel a {current.state} job")
        state_name = "cancelled" if current.state == "queued" else current.state
        return update_job(job_id, cancel_requested=True, state=state_name)

    @application.post("/jobs/{job_id}/retry", response_model=JobRecord, status_code=202)
    def retry_job(job_id: str, background_tasks: BackgroundTasks) -> JobRecord:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        current = jobs[job_id]
        if current.state not in {"failed", "cancelled"}:
            raise HTTPException(status_code=409, detail="Only failed or cancelled jobs can be retried")
        updated = update_job(job_id, state="queued", progress=0, error=None, cancel_requested=False)
        background_tasks.add_task(process_job, job_id)
        return updated

    @application.get("/settings", response_model=RuntimeSettings)
    def settings() -> RuntimeSettings:
        config = get_pipeline().config
        return RuntimeSettings(
            final_k=config.retrieval.top_k,
            candidate_k=config.retrieval.candidate_k,
            fusion=cast(Literal["rrf", "weighted"], config.retrieval.fusion),
            rrf_k=config.retrieval.rrf_k,
            dense_weight=config.retrieval.dense_weight,
            sparse_weight=config.retrieval.sparse_weight,
            provider=config.generation.provider,
            model=config.generation.model,
            vision=config.generation.use_vision,
            demo_mode=config.generation.provider == "mock",
        )

    @application.put("/settings", response_model=RuntimeSettings)
    def update_settings(request: RuntimeSettings) -> RuntimeSettings:
        if request.dense_weight + request.sparse_weight <= 0:
            raise HTTPException(status_code=422, detail="At least one retrieval weight must be positive")
        provider = "mock" if request.demo_mode else request.provider
        current = get_pipeline()
        retrieval = current.config.retrieval.model_copy(
            update={
                "top_k": request.final_k,
                "candidate_k": request.candidate_k,
                "fusion": request.fusion,
                "rrf_k": request.rrf_k,
                "dense_weight": request.dense_weight,
                "sparse_weight": request.sparse_weight,
            }
        )
        generation = current.config.generation.model_copy(
            update={"provider": provider, "model": request.model, "use_vision": request.vision}
        )
        try:
            state["pipeline"] = MMRAGPipeline(
                current.config.model_copy(update={"retrieval": retrieval, "generation": generation}), current.chunks
            )
        except LLMError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return settings()

    @application.get("/history", response_model=list[HistoryEntry])
    def history(search: str = "") -> list[HistoryEntry]:
        return get_history().list(search)

    @application.patch("/history/{history_id}", response_model=HistoryEntry)
    def update_history(history_id: str, request: HistoryUpdate) -> HistoryEntry:
        entry = get_history().update(history_id, title=request.title, pinned=request.pinned)
        if entry is None:
            raise HTTPException(status_code=404, detail="History entry not found")
        return entry

    @application.post("/history/{history_id}/duplicate", response_model=HistoryEntry)
    def duplicate_history(history_id: str) -> HistoryEntry:
        entry = get_history().get(history_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="History entry not found")
        return get_history().add(entry.question, entry.payload, title=f"{entry.title} (copy)")

    @application.delete("/history/{history_id}", status_code=204)
    def delete_history(history_id: str, confirm: bool = False) -> Response:
        if not confirm:
            raise HTTPException(status_code=409, detail="Deletion requires confirm=true")
        if not get_history().delete(history_id):
            raise HTTPException(status_code=404, detail="History entry not found")
        return Response(status_code=204)

    @application.delete("/history", status_code=204)
    def clear_history(confirm: bool = False) -> Response:
        if not confirm:
            raise HTTPException(status_code=409, detail="Clear history requires confirm=true")
        get_history().clear()
        return Response(status_code=204)

    @application.get("/evaluations")
    def evaluations() -> list[dict[str, object]]:
        reports = get_pipeline().config.paths.reports
        output: list[dict[str, object]] = []
        for metrics_path in sorted(reports.glob("*/metrics.json"), reverse=True):
            run_dir = metrics_path.parent
            output.append({"run_id": run_dir.name, "metrics": json.loads(metrics_path.read_text(encoding="utf-8"))})
        return output

    @application.get("/evaluations/{run_id}")
    def evaluation(run_id: str) -> dict[str, object]:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise HTTPException(status_code=404, detail="Evaluation not found")
        report_dir = get_pipeline().config.paths.reports / run_id
        if approved_path(report_dir, get_pipeline().config.paths.reports) is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        metrics = report_dir / "metrics.json"
        if not metrics.is_file():
            raise HTTPException(status_code=404, detail="Evaluation not found")
        result: dict[str, object] = {"run_id": run_id, "metrics": json.loads(metrics.read_text(encoding="utf-8"))}
        summary = report_dir / "summary.md"
        if summary.is_file():
            result["summary"] = summary.read_text(encoding="utf-8")
        for name in ("config.json", "environment.json"):
            artifact = report_dir / name
            if artifact.is_file():
                result[name.removesuffix(".json")] = json.loads(artifact.read_text(encoding="utf-8"))
        git_sha = report_dir / "git_sha"
        if git_sha.is_file():
            result["git_sha"] = git_sha.read_text(encoding="utf-8").strip()
        per_question = report_dir / "per_question.jsonl"
        if per_question.is_file():
            result["per_question"] = [
                json.loads(line) for line in per_question.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
        return result

    @application.get("/figure/{chunk_id}")
    def figure(chunk_id: str) -> Response:
        chunk = get_pipeline().chunk(chunk_id)
        if chunk is None or chunk.modality is not Modality.FIGURE or chunk.image_path is None:
            raise HTTPException(status_code=404, detail="Figure not found")
        path = approved_path(chunk.image_path, get_pipeline().config.paths.processed)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Figure file missing")
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return Response(path.read_bytes(), media_type=media_type)

    @application.get("/page/{doc_id}/{page}.png")
    def page_image(doc_id: str, page: int) -> Response:
        if page < 1:
            raise HTTPException(status_code=404, detail="Page not found")
        chunk = next((item for item in get_pipeline().chunks if item.doc_id == doc_id), None)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Document not found")
        source_path = approved_path(
            chunk.source_path,
            get_pipeline().config.paths.raw,
            get_pipeline().config.paths.processed,
        )
        if source_path is None or not source_path.is_file():
            raise HTTPException(status_code=404, detail="Document file missing")
        try:
            import pymupdf

            with pymupdf.open(source_path) as document:  # type: ignore[no-untyped-call]
                if page > document.page_count:
                    raise HTTPException(status_code=404, detail="Page not found")
                pixmap = document[page - 1].get_pixmap(dpi=140, alpha=False)
                payload = io.BytesIO(pixmap.tobytes("png")).getvalue()
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"Unable to render page: {error}") from error
        return Response(payload, media_type="image/png")

    @application.get("/page/{doc_id}/{page}/metadata")
    def page_metadata(doc_id: str, page: int) -> dict[str, object]:
        if page < 1:
            raise HTTPException(status_code=404, detail="Page not found")
        chunk = next((item for item in get_pipeline().chunks if item.doc_id == doc_id), None)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Document not found")
        source = approved_path(
            chunk.source_path,
            get_pipeline().config.paths.raw,
            get_pipeline().config.paths.processed,
        )
        if source is None or not source.is_file():
            raise HTTPException(status_code=404, detail="Document file missing")
        import pymupdf

        with pymupdf.open(source) as document:  # type: ignore[no-untyped-call]
            if page > document.page_count:
                raise HTTPException(status_code=404, detail="Page not found")
            pdf_page = document[page - 1]
            return {
                "document": doc_id,
                "page": page,
                "page_count": document.page_count,
                "width": float(pdf_page.rect.width),
                "height": float(pdf_page.rect.height),
                "rotation": int(pdf_page.rotation),
            }

    return application
