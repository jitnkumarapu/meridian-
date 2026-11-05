# Master Prompt: Complete and Improve the Multimodal Finance RAG Product

## Mission

Continue work directly in the existing repository and turn the current verified prototype into a
reliable, measurable, production-quality multimodal RAG system for financial reports.

Work in small increments. Run commands and tests before claiming anything is complete. Never invent
metrics, answers, reports, model results, corpus files, human verification, or provider availability.
Preserve the offline workflow while adding production capabilities.

The project is a research prototype and must keep the visible disclaimer: **not financial advice**.

---

## Verified current state

Repository root:

```text
C:\Users\jitin\OneDrive\Documents\ChatGPT\minor spec
```

Implemented and verified:

- PDF ingestion with PyMuPDF.
- OCR interface with Tesseract support.
- Text, table, and figure extraction.
- Structured table grids and serialized table content.
- Figure captions, image paths, and bounding boxes.
- Separate text/table dense retrieval and figure retrieval pathways.
- BM25 sparse retrieval.
- Reciprocal Rank Fusion and weighted fusion support.
- Deterministic hash encoder for offline tests.
- Sentence Transformers production adapter.
- OpenCLIP production adapter.
- Citation parsing and citation validation.
- Grounding gate with one strict retry and abstention.
- Deterministic `MockLLM` for offline CI.
- OpenAI-compatible, Anthropic, Ollama, and Gemini provider abstractions.
- Gemini provider with text and image evidence support.
- Prompt hashing.
- Fair text-only baseline that removes figures and flattens tables.
- Evaluation metrics, confidence intervals, and saved reports.
- FastAPI retrieval, answer, health, document, figure, and page endpoints.
- React evidence-search interface.
- Streamlit compatibility interface.
- Synthetic fixture generation.
- Docker and configuration scaffolding.
- Path-confinement security checks for figure and page serving.
- Automatic `.env` loading through `python-dotenv`.
- `.env` is ignored by Git.

Current production configuration:

```yaml
generation:
  provider: gemini
  model: gemini-3.5-flash-lite
  temperature: 0
  max_tokens: 2048
  use_vision: true
```

The local Gemini key is stored in the ignored repository-root `.env` file under:

```text
GEMINI_API_KEY=...
```

Never read, print, copy, commit, or expose the value. Never put it in frontend code, YAML, logs,
reports, screenshots, or documentation.

Verified checks:

- Python tests: 33 passed.
- Ruff lint: passed.
- Ruff formatting: passed.
- Strict mypy: passed.
- Frontend tests: passed.
- Frontend production build: passed.
- Tesseract 5.5.3 installed and verified.
- `sentence-transformers/all-MiniLM-L6-v2` downloaded and loaded on CPU.
- OpenCLIP `ViT-B-32` with `laion2b_s34b_b79k` downloaded and loaded on CPU.
- Real Gemini request completed successfully with a grounded answer and valid citation.
- Synthetic fixture ingestion produced text, table, figure, and OCR evidence.
- Multimodal and baseline indexes were built.

Existing measured fixture results are synthetic only and must not be presented as real-world quality:

- Previous fixture report: `reports/20261103T204455Z`.
- Three unverified questions.
- Baseline Recall@5: 0.667.
- Multimodal Recall@5: 1.000.
- Baseline MRR: 0.667.
- Multimodal MRR: 0.750.
- Citation validity: 1.000.

---

## Current limitations and unfinished work

Do not claim these are complete:

1. The 15 real annual-report PDFs are not yet present.
2. The corpus manifest contains official landing pages requiring manual PDF selection.
3. No human-verified 60-question gold set exists.
4. No real-corpus headline metrics exist.
5. Deterministic financial value parsing and calculation traces are incomplete.
6. Table extraction quality scoring and extractor selection are incomplete.
7. Figure filtering does not yet robustly reject logos, icons, and decorative images.
8. Retrieval indexes lack complete compatibility metadata and cache invalidation.
9. Retrieval explanations and tuning/development splits are incomplete.
10. The frontend is not yet the complete multi-page product requested in the original specification.
11. Upload jobs, progress, cancellation, retry, and isolation are incomplete.
12. Saved history, document library, evaluation UI, and settings UI are incomplete.
13. PDF region overlay and responsive drawer behavior are incomplete.
14. Playwright, accessibility, mobile, and full API integration coverage are incomplete.
15. Real scanned annual-report OCR has not been manually verified.
16. Docker clean-environment verification remains pending.

---

## Non-negotiable rules

1. Read the README, architecture, evaluation, deviations, results, configuration, tests, and source
   before editing.
2. Preserve all existing API contracts unless a versioned migration is justified.
3. Keep the offline fixture path network-free and GPU-free.
4. Keep MockLLM and deterministic encoders for tests only.
5. Never silently fall back from a configured production provider to MockLLM.
6. Use pretrained encoders only; do not train or fine-tune encoders.
7. Never compare raw scores from incompatible retrieval systems.
8. Fuse ranked lists with RRF or independently normalized weighted fusion.
9. Preserve complete chunk metadata and trace every answer to retrieved chunks.
10. Validate every generated citation against retrieved evidence.
11. Retry invalid citations once, then return `Insufficient evidence`.
12. Do not claim exact chart values unless present in caption, OCR, nearby text, or a table.
13. Use Python `Decimal` for financial arithmetic, never binary floating point.
14. Do not tune on the locked test set.
15. Every displayed metric must come from a saved evaluation run.
16. Add a regression test for every bug fixed.
17. Do not commit `.env`, keys, tokens, private PDFs, or secrets.
18. Use parameterized queries and approved-root path validation.
19. Surface errors explicitly; do not use broad silent catches or success-shaped fallbacks.
20. Prefer free/open-source tools and local execution. Use Gemini only when a configured key is
    available and clearly label external-provider evidence handling.

---

## Preferred free/open-source stack

Use existing dependencies first. Add dependencies only when needed and update `pyproject.toml` and
`uv.lock` together.

- Retrieval embeddings: Sentence Transformers, initially `all-MiniLM-L6-v2`; benchmark BGE or
  multilingual alternatives only on a development split.
- Figure embeddings: OpenCLIP with a separate image/text embedding space.
- Sparse retrieval: existing finance-aware BM25.
- Optional reranker: a small pretrained cross-encoder such as
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, behind a configuration flag.
- Local generation fallback: Ollama with an installed open model, but never silently selected.
- Free hosted generation: Gemini only when explicitly configured through `GEMINI_API_KEY`.
- Orchestration: use LangGraph only for explicit stateful workflow orchestration if it materially
  improves retries, branching, streaming, or observability. Do not rewrite extraction/retrieval
  merely to use LangChain.
- Prompt/provider wrappers: LangChain may be used only where it reduces provider-specific code and
  preserves current schemas and citation validation.
- OCR: Tesseract.
- Tables: PyMuPDF and pdfplumber comparison.
- API: FastAPI.
- Frontend: React and TypeScript.
- Testing: pytest, Ruff, mypy, Vitest, Playwright.
- Tracing: structured local JSON logs; use OpenTelemetry only if it can remain optional and
  network-free in CI.

---

## Required execution order

### Phase A: Re-establish the baseline

Run:

```powershell
uv sync --extra models --extra ocr --extra tables --extra app --extra eval --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/mmrag
npm test --prefix frontend
npm run build --prefix frontend
```

Fix only regressions caused by current work. Record exact results.

### Phase B: Harden configuration and providers

- Confirm `.env` is loaded from the repository root.
- Keep `.env` ignored.
- Redact API keys from logs and exception text.
- Add provider health checks for missing keys, unavailable model, timeout, malformed response,
  unsupported vision, and rate limits.
- Add retry/backoff only for safe transient failures.
- Keep Gemini, Ollama, and MockLLM explicit.
- Add provider/model/latency/token/retry/grounding metadata to answer logs.
- Add tests proving no key appears in logs, URLs, errors, or serialized answers.

### Phase C: Improve answer generation

- Version prompts and retain prompt hashes.
- Separate question, retrieved evidence, grounding rules, citation rules, numerical rules, and
  output format.
- Require exact canonical citations: `[document-id, p.12, modality]`.
- Use structured output where supported, but keep the existing citation gate authoritative.
- Add compact and detailed answer styles.
- Send only retrieved figure images to vision providers.
- Never send unrelated pages or raw uploaded files.
- Add a LangGraph workflow only if useful:

```text
normalize -> retrieve -> evidence_gate -> classify
  -> numerical_calculation
  -> visual_generation
  -> grounded_generation
  -> citation_validation
  -> strict_retry_once
  -> abstain_or_return
```

The graph state must contain the original question, retrieved evidence, provider metadata,
calculation trace, generated text, validation result, retry count, and final answer. Preserve the
existing `Answer` schema and MockLLM tests.

### Phase D: Add deterministic financial reasoning

Implement typed financial values containing:

- Original display text
- `Decimal` numeric value
- Currency
- Scale
- Unit
- Percentage flag
- Fiscal period
- Row and column labels
- Source chunk ID
- Document and page

Support:

- Currency symbols and codes
- Commas and decimals
- Percentages
- Parentheses for negatives
- Basis points
- Thousands, millions, billions
- Lakhs and crores

Support safe calculations:

- Difference
- Percentage change
- Ratio
- Sum
- Average
- Minimum
- Maximum

Require every operand to come from retrieved evidence. Reject ambiguous units, incompatible
periods, missing values, and division by zero. Return an arithmetic trace with operands, formula,
normalized units, result, and citations. Add tests before and after implementation.

### Phase E: Improve retrieval

- Add query normalization for fiscal years, percentages, currencies, abbreviations, and number
  scales while preserving the original query.
- Add conservative synonym expansion and log expansions.
- Add document, modality, and fiscal-year filters.
- Add optional cross-encoder reranking for text/table chunks.
- Add retrieval explanation objects:
  dense rank/score, BM25 rank/score, figure rank/score, fusion contribution, reranker score,
  filters, and final rank.
- Add index metadata:
  encoder IDs, checkpoints, dimensions, normalization, preprocessing version, corpus checksum,
  config hash, and build time.
- Refuse incompatible index loads.
- Add content/model-identity embedding cache with invalidation.
- Tune candidate K, final K, and fusion settings only on a development set.
- Compare dense, BM25, hybrid RRF, weighted fusion, and reranking using saved reports.

### Phase F: Improve extraction

Tables:

- Compare PyMuPDF and pdfplumber.
- Score non-empty cells, headers, alignment, row consistency, blanks, repeats, and warnings.
- Detect nearby titles, units, currency, scale, periods, and continuation headers.
- Preserve exact display strings and normalized values separately.
- Add review artifacts with page crop, grid, serialization, extractor, score, and warnings.

Figures:

- Reject repeated logos, tiny icons, headers, footers, near-blank images, and decorative photos.
- Detect scanned pages as OCR pages before creating figure chunks.
- Link captions and nearby text with bounding-box distance and reading order.
- Store OCR, caption, context, extraction method, quality score, and decision.
- Keep one chunk ID across image, text, and BM25 signals.

### Phase G: Build the real corpus

Use [data/manifests/corpus.csv](data/manifests/corpus.csv). For each of the 15 reports:

- Download the actual official PDF manually from the listed source.
- Reject HTML pages renamed `.pdf`.
- Record retrieval date, checksum, page count, source terms, and status.
- Do not bypass login, robots, or access controls.
- Inspect representative text, tables, figures, scanned pages, and OCR.

Required human action if files are missing:

```text
Place the exact manifest-named PDFs in data/raw and update corpus.csv.
```

### Phase H: Evaluation

Create at least 60 human-verified questions balanced across text, table, figure, and multimodal
questions. Keep development and locked test splits separate.

Every question needs:

- ID
- Question
- Reference answer
- Gold document/page/modality/chunk IDs
- Units and period if applicable
- Question type
- Author
- Reviewer
- Verified flag

Report Recall@1/3/5/10, Precision@1/3/5/10, MRR, citation validity, groundedness, numerical exact
match, latency, token usage, confidence intervals, sample sizes, and failure categories.

Compare:

- Baseline vs multimodal
- Dense vs BM25 vs hybrid
- RRF vs weighted fusion
- Text vs text+tables vs text+tables+figures
- Reranker off vs on

Never publish a real-corpus headline without verified questions.

### Phase I: Complete the product

Implement and test:

- Document library with processing state and counts.
- Upload validation and isolated idempotent jobs.
- Progress polling or SSE with cancellation and retry.
- Search filters and compare mode.
- Evaluation results page reading only `reports/<run_id>`.
- Saved question history without secrets or raw files.
- PDF page drawer with scaled bounding boxes, rotation, zoom, and no-region state.
- Settings for K, fusion, modality, provider, model, vision, and demo mode.
- Mobile layout from 360px upward.
- Keyboard navigation, focus states, ARIA labels, contrast, reduced motion.
- Playwright flows with no console errors, failed API calls, broken images, or accessibility failures.

### Phase J: Final verification

Run all tests, builds, fixture workflows, API checks, Docker checks, and any real-corpus checks
that are actually possible. Record machine details and exact run IDs. Update documentation only
with measured evidence.

---

## Required deliverables

At the end, report:

1. Files changed.
2. Commands run.
3. Pass/fail result for every command.
4. Measured acceptance evidence.
5. Saved evaluation run IDs.
6. Provider/model/index versions.
7. Known limitations.
8. Human actions still required.
9. Privacy, licensing, and external-provider notes.

Do not state “complete” unless the corresponding acceptance checks actually passed.
