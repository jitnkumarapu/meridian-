# Implementation status

## Completed and verified

- 15/15 manifest PDFs are present and valid (`ready=15 missing=0 invalid=0`).
- Real extraction completed for 3,868 pages and produced 14,145 evidence chunks.
- Text, table, figure, OCR metadata, page provenance, and extraction warnings are persisted under
  `data/real_processed`.
- Multimodal and flattened baseline indexes are built under `data/real_indexes`.
- Backend: 63 tests pass; Ruff and strict mypy pass.
- Frontend: 7 tests pass, the production Vite build passes, and 4 Playwright checks pass
  across desktop and 360px mobile, including axe accessibility checks.
- Extraction audit: `reports/real/extraction_review.json` reports `status=pass`.
- Tesseract 5.5.3 is installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` and a real
  OCR smoke test passed with 80.4% confidence.
- The 60 real-corpus questions are owner-confirmed and recorded as verified with reviewer and
  review date metadata. A retrieval-only run is saved under `reports/real/retrieval_verified`.
- API conversations now preserve a bounded three-turn context using a `conversation_id`.
- Live end-to-end verification succeeded through the browser proxy: `/api/documents` returns all
  15 documents, and a real Gemini `/ask` request returned HTTP 200 with a grounded answer and
  valid citation.
- The API and React UI have been smoke-tested on ports 8010 and 5175.

## Remaining environment-dependent gates

- A full real Gemini answer-quality evaluation remains environment-dependent. The production model
  is now `gemini-3.5-flash-lite`; a live smoke request completed with a grounded answer and valid
  citations after transient HTTP 503 responses were retried. No partial metrics are treated as a
  completed full evaluation; the benchmark should be rerun when the provider is stable.
- Docker verification is intentionally deferred. Docker Desktop is installed, but its Linux
  engine currently returns HTTP 500 on the Docker API.

## Current local run

- API: `http://127.0.0.1:8010`
- UI: `http://127.0.0.1:5177`
- Health: `http://127.0.0.1:8010/health`
