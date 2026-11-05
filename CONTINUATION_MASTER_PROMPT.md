# Continuation Master Prompt: Finish and Validate the Real-Corpus Multimodal Finance RAG Product

## How to use this prompt

Give this entire file to the next implementation agent after placing the required PDFs in the
workspace. Treat it as execution instructions. Cross-check it against both existing specifications:

- `MULTIMODAL_RAG_PRODUCT_IMPLEMENTATION_AGENT_PROMPT.md`
- `NEXT_PHASE_MASTER_PROMPT.md`

Instructions in those documents describe product requirements; they are not evidence that a task is
complete. Inspect the repository and executed artifacts before making any completion claim.

## Mission

Continue from the current implementation. Do not rebuild working components unnecessarily. Finish
the real-corpus and human-evaluation stages, correct any defect revealed by real data, verify the real
provider and OCR paths when authorized, execute all comparisons, and leave an evidence-backed final
handoff. Never invent documents, reviews, metrics, screenshots, provider results, or Docker results.

## Workspace

Expected repository root on the current machine:

```text
C:\Users\jitin\OneDrive\Documents\ChatGPT\minor spec
```

Preserve existing user files and unrelated changes. The repository began in an all-untracked state,
so do not assume `git diff` represents the original baseline. Never print, read back, or commit API
keys. Never silently fall back from a requested production provider or encoder to a mock/hash model.

## Current verified implementation

### Core architecture

- Typed YAML configuration for paths, ingestion, embedding, retrieval, generation, and baseline.
- PyMuPDF ingestion with OCR fallback hooks, structured chunk schemas, provenance, checksums, and
  content-hash caching.
- Separate text/table dense space and OpenCLIP figure space. Scores from incompatible spaces are
  fused by rank or normalized weighted fusion and are never directly compared as raw similarities.
- Persistent dense stores, number-aware BM25, embedding cache, index metadata, vector checksums,
  checkpoint IDs, configuration hashes, and incompatible-index rejection.
- Optional CrossEncoder reranker and retrieval explanations with source ranks and fusion contribution.
- Baseline transformation uses the same questions while flattening tables and excluding figures.

### Extraction

- Text chunking with headings, page provenance, and OCR metadata.
- Competing PyMuPDF and pdfplumber table candidates, quality scoring, extractor selection, normalized
  cells, title/unit/period context, table crops, review metadata, and quality warnings.
- Figure filtering for repeated headers/logos, tiny images, near-blank images, decorative images, and
  scanned-page false positives.
- Embedded-image and basic vector-region extraction with captions, nearby text, OCR fields, crops,
  quality decisions, and figure-review artifacts.

### Search and generation

- Finance-aware query normalization, fiscal-year/document/modality filters, dense/BM25/hybrid search,
  RRF and weighted fusion, diversity handling, and explainable results.
- Structured prompt v3, compact/detailed styles, selected-evidence-only vision attachments, bounded
  transient retries, provider diagnostics, secret-free errors, token/latency/retry logging, and no
  missing-credential mock fallback.
- Citation resolution, one strict repair retry, rejection of unsupported/no-citation answers, and
  `Insufficient evidence` behavior.
- Conservative deterministic financial reasoning using `Decimal`, currency/unit/period provenance,
  percentages, basis points, thousands/lakhs/millions/crores/billions, difference, percentage change,
  ratio, sum, average, min/max, formulas, operand citations, and ambiguity/division safety.

### API and product

- FastAPI endpoints for ask, provisional SSE response delivery, documents, page images/metadata,
  figures, upload jobs, job polling/cancel/retry, reindex/reingest/remove, settings, history,
  diagnostics, and saved evaluation artifacts.
- Upload size/page/PDF validation, filename sanitization, path confinement, checksum idempotence, and
  explicit job stages/warnings/errors.
- SQLite-backed local question history with parameterized queries, search, pin, rename, duplicate,
  delete, clear, and curated payloads that exclude keys/raw files/prompts.
- React/Vite UI with search, modality/document filters, baseline comparison, cancellation, duplicate
  prevention, skeleton/error/empty states, evidence cards, tables/figures, exact scaled bounding boxes,
  page navigation, zoom/rotation, document review, job actions, saved evaluations, history actions,
  runtime settings, dark/light themes, focus states, ARIA, reduced motion, and 360px layout.
- Streamlit entry point remains available as a secondary interface.

### Evaluation

- Recall and precision at 1/3/5/10, MRR, citation validity, grounded rate, numerical exact match,
  latency, token usage, bootstrap confidence intervals, paired significance, sample sizes, breakdowns,
  failure categories, per-question JSONL, config, environment, git SHA, summary, and SVG plots.
- Deterministic development/locked-test split builder for at least 60 independently reviewed questions.
- Benchmark download/provenance scaffolding in `scripts/download_benchmarks.py`.

## Important fixes made in the latest pass

- Decoupled embedding choice from answer-provider choice. A mock generator no longer forces hash
  embeddings; only `embedding.test_mode: true` selects the hash encoder.
- Built and reloaded a real fixture index using `sentence-transformers/all-MiniLM-L6-v2` at 384
  dimensions and `open_clip/ViT-B-32/laion2b_s34b_b79k`.
- Added content/model-keyed embedding caching and persisted production checkpoint identities.
- Added Playwright and axe acceptance tests and corrected mobile navigation and dark-theme contrast.
- Prevented Vitest from collecting Playwright specifications.
- Exposed history rename/clear, job cancel/retry, and document reindex/reingest/remove in the React UI.
- Fixed the human gold-set builder so its rich authoring schema is normalized to the evaluator's
  `EvaluationCase` schema before writing dev/test files.
- Added `scripts/verify_corpus.py` to validate exact manifest filenames and record checksums, page
  counts, retrieval dates, and status.

## Existing executed evidence

At the end of the previous pass:

- Ruff lint and format checks passed.
- Strict mypy passed for 42 source files.
- Fifty-seven Python tests passed with approximately 82% statement coverage.
- Seven frontend component tests passed.
- The TypeScript/Vite production build passed.
- Four Playwright tests passed across desktop and 360px mobile, including axe and console-error checks.
- Docker and Tesseract were not installed/on `PATH` on this host.

Saved synthetic runs include:

- `fixture-complete-20261105`: deterministic hash-encoder plumbing run.
- `fixture-production-20261105`: production MiniLM/OpenCLIP fixture run.

For `fixture-production-20261105`, baseline Recall@5/MRR were 0.667/0.667 and multimodal
Recall@5/MRR were 1.000/0.511, with citation validity 1.000. This run has only three synthetic,
unverified questions and must never be presented as real-corpus quality.

## User's required manual work

Follow `docs/MANUAL_CORPUS_AND_GOLD_SET_GUIDE.md`. In summary:

1. Do not redownload the corpus: all 15 official PDFs are already in `data/raw` with checksums and
   page counts recorded in `data/manifests/corpus.csv`.
2. If a file is later lost or corrupted, run `uv run python scripts/download_corpus.py`, followed by
   `uv run python scripts/verify_corpus.py`.
3. Spot-check company, fiscal year, report type, and completeness while reviewing source pages.
4. After ingestion, inspect representative text, tables, figures, and scanned pages.
5. Write at least 60 questions, preferably 15 each for text/table/figure/multimodal, covering all
   reports and using exact pages/modalities/chunk IDs.
6. Have a different person review every question against the original PDF. Set `verified: true` only
   after the answer, units, period, pages, modalities, and chunks are confirmed.
7. Keep the locked test split hidden from tuning decisions.

The exact required filenames are:

```text
infosys-2024.pdf
tcs-2024.pdf
wipro-2024.pdf
hcltech-2024.pdf
reliance-2024.pdf
hdfc-bank-2024.pdf
itc-2024.pdf
asian-paints-2024.pdf
apple-2024.pdf
microsoft-2024.pdf
nvidia-2024.pdf
alphabet-2023.pdf
amazon-2023.pdf
intel-2023.pdf
cisco-2024.pdf
```

## Next agent execution order

### 1. Re-establish the baseline

Run and record exact results:

```powershell
uv sync --extra dev --extra app --extra eval --extra tables --extra models
uv run ruff check .
uv run ruff format --check .
uv run mypy src/mmrag
uv run pytest
npm test --prefix frontend
npm run build --prefix frontend
npm run e2e --prefix frontend
```

Do not weaken tests or coverage thresholds to obtain a pass.

### 2. Verify and isolate the real corpus

Run:

```powershell
uv run python scripts/verify_corpus.py
```

The expected result is `ready=15 missing=0 invalid=0 total=15`; this was achieved on 2026-11-05.
Stop if a later run reports any missing, invalid, or wrongly named file. Keep the synthetic
fixture out of real-corpus indexes and reports. Use a separate real-corpus processed/index path or,
with explicit user approval, clean only the known generated synthetic processed/index artifacts.
Never recursively delete an unresolved or broad path.

### 3. Ingest and manually inspect

```powershell
uv run mmrag ingest data/raw --manifest data/manifests/corpus.csv --config configs/production.yaml --force
uv run mmrag index --config configs/production.yaml
uv run mmrag index --config configs/production.yaml --baseline
```

Inspect ingestion reports and review artifacts. Use the UI to inspect every report. Pay special
attention to multi-page tables, merged headers, units, continued tables, repeated logos, vector
charts, full-page scans, OCR confidence, rotated pages, and bounding-box accuracy. Fix general code,
not one-off hard-coded answers. Add regression fixtures/tests for each discovered failure.

### 4. Validate the 60-question set

The user-authored input should be:

```text
data/eval/own_corpus_questions.jsonl
```

Run:

```powershell
uv run python scripts/build_gold_set.py data/eval/own_corpus_questions.jsonl
```

Require at least 60 records, at least 12 of each question type, complete author/reviewer fields, and
`verified: true`. Confirm the normalized outputs load through `EvaluationCase`. Do not change gold
answers merely to agree with system output.

### 5. Tune only on development questions

Run the development split first. Investigate retrieval misses, extraction errors, wrong modality,
wrong period/unit, calculation errors, unsupported claims, citation errors, OCR errors, and provider
failures. Add tests for every code defect. Do not inspect locked-test aggregate/per-question output
while tuning.

Required comparisons on identical questions:

- baseline versus multimodal;
- dense-only versus BM25-only versus hybrid;
- RRF versus normalized weighted fusion;
- text versus text+tables versus text+tables+figures;
- reranker disabled versus enabled.

If these comparisons are not yet exposed as one reproducible evaluation command, implement an
ablation runner that writes each system's exact config and per-question records under one run.

### 6. Run external benchmarks responsibly

Use official FinQA and TAT-QA contexts/supporting labels and small seeded DocVQA/SlideVQA subsets.
Record dataset version, URL, retrieval date, checksum, licence/terms, seed, and sample IDs. Keep their
metrics separate from the 15-report corpus. Do not imply FinQA/TAT-QA measure visual retrieval.

### 7. Verify OCR, provider, and streaming

- If Tesseract is installed, run OCR on at least one real scanned page and manually compare output to
  the page image. Record engine version and review evidence.
- With explicit authorization and a configured environment variable, run the real provider on text,
  table, figure, and multimodal questions. Do not print the credential or send unrelated pages.
- Confirm missing credentials fail explicitly.
- Audit streaming carefully: provisional text must not be labeled grounded; final citations must be
  validated before replacement. The existing SSE contract should be tested against true upstream
  provider streaming when supported, not merely post-completion chunking.

### 8. Reliability and security audit

Recheck upload isolation, idempotence, cancellation during active processing, retry behavior, partial
output cleanup, restart behavior, CORS, served-path confinement, PDF/page limits, filename handling,
SQLite parameterization, secret redaction, request IDs, and structured logs. Add concurrency and
failure-injection tests where the real corpus exposes slow paths.

### 9. Product and browser verification

Run the product and test search, compare, evidence opening, exact page highlight, filters, upload,
cancel/retry, document review/actions, evaluation tabs/failures, history CRUD/export, settings,
offline/API failures, light/dark themes, keyboard-only use, reduced motion, and 360px layout. Keep
axe violations, console errors, failed requests, and broken images at zero in the tested flows.

### 10. Locked test and final reporting

Only after development is frozen:

```powershell
uv run mmrag eval --gold data/eval/own_corpus_locked_test.jsonl --run-id real-locked-YYYYMMDD
```

Publish exact run IDs, sample sizes, confidence intervals, failure counts, configs, environment, model
checkpoint IDs, and limitations. Never replace a measured value with a target value.

### 11. Docker verification

On a Docker-capable host, build from a clean context and execute ingestion, index loading, one query,
API health, and frontend smoke tests. Record commands and image identifiers. If Docker remains
unavailable, keep the gate explicitly blocked.

## Files that define truth

- Corpus manifest: `data/manifests/corpus.csv`
- Manual workflow: `docs/MANUAL_CORPUS_AND_GOLD_SET_GUIDE.md`
- Human authoring example: `data/eval/own_corpus_questions.example.jsonl`
- Gold builder: `scripts/build_gold_set.py`
- Corpus verifier: `scripts/verify_corpus.py`
- Main configuration: `configs/default.yaml`
- Production provider configuration: `configs/production.yaml`
- Implementation matrix: `docs/implementation_status.md`
- Evaluation protocol: `docs/evaluation.md`
- Deviations: `docs/deviations.md`
- Results: `docs/results.md`

## Non-negotiable rules

- No fabricated metrics, reviews, documents, OCR outcomes, provider outcomes, or screenshots.
- No headline results from synthetic or unverified questions.
- Never tune on the locked test set.
- Never expose or commit secrets.
- Never silently substitute mock/hash components for configured production components.
- Never compare raw MiniLM and CLIP similarity scores directly.
- Every accepted claim must resolve to retrieved evidence.
- Every deterministic calculation must retain operands, units, formula, and citations.
- Preserve original PDFs and provenance.
- Document every unresolved deviation.

## Definition of completion

Completion requires all 15 verified official reports, a traceable manifest, real OCR review, manual
extraction review, production index build/reload, authorized real-provider validation, at least 60
independently reviewed questions, frozen locked-test results, required ablations, external benchmark
provenance, all Python/frontend/browser checks, clean Docker execution, no secret leakage, no fake or
placeholder results, and an updated evidence-backed handoff.

If any requirement depends on missing user data, software, credentials, authorization, or independent
human judgment, report it as blocked and state the smallest exact action needed from the user.
