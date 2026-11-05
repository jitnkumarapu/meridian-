# ADR-0001: explicit modality spaces and a lightweight core

Status: accepted, 2026-11-03.

## Decision

Use Python 3.11 with Pydantic contracts, PyMuPDF extraction, Sentence Transformers for text and
table embeddings, OpenCLIP for image/query embeddings, exact inner-product search at this corpus
size, and BM25. Text/table and image embeddings remain in separate spaces and are combined only by
rank fusion. FastAPI is the system boundary; React is the primary UI and Streamlit is a thin client.

The CI core has deterministic test encoders and `MockLLM`, selected explicitly by test configuration.
They prevent network/model downloads in CI but are never represented as pretrained production models.
Heavy model, OCR, table, app, and evaluation packages are optional dependency groups.

## Consequences

The persisted index contains encoder identity and config hash, so a mismatched model cannot silently
load it. Figure text participates in the text and BM25 indexes using the same figure chunk ID. Raw
scores from different spaces are never compared. Production startup fails clearly when a selected
provider or encoder dependency is unavailable.

Version ranges and API choices were checked against official project documentation on this date;
exact resolved versions are captured by `uv.lock` when setup runs.
