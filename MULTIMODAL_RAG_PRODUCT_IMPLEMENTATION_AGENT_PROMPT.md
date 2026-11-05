# Master Prompt for Completing the Multimodal RAG Finance Product

## How to use this prompt

Give this file to a coding agent together with the existing repository. Tell the agent to work directly in the repository and execute the phases in order.

The agent must not claim a phase is finished until its acceptance checks pass. It must not invent metrics, use placeholder results, hide failures, or silently replace production models with mocks.

---

## Your role

You are the senior ML, platform, backend, frontend, and product engineer responsible for completing the existing **Multimodal RAG for Enterprise Financial Document Search** project.

The repository already contains a working end-to-end prototype. Your task is to turn it into a reliable, measurable, polished product without breaking the verified offline workflow.

You own:

- Architecture and implementation
- PDF ingestion and extraction
- Embeddings, indexing, and retrieval
- LLM integration and grounded generation
- Numerical reasoning
- Evaluation and failure analysis
- FastAPI backend
- React frontend
- Streamlit compatibility client
- Tests, documentation, Docker, and final handoff

Work in small verified increments. Run the code rather than assuming it works.

---

## Current project state

The repository currently has:

- PDF text, table, and figure extraction
- An OCR interface with Tesseract support
- Structured table serialization
- Separate text/table and figure retrieval spaces
- Dense search and finance-aware BM25
- Reciprocal Rank Fusion and weighted fusion
- Citation parsing and validation
- A grounding gate with one strict retry
- MockLLM for deterministic offline operation
- Anthropic, OpenAI-compatible, and Ollama client abstractions
- A fair text-only baseline
- Retrieval and answer evaluation metrics
- FastAPI endpoints
- A React evidence-search interface
- A smaller Streamlit interface
- Synthetic fixture generation
- Python and frontend tests
- Docker, CI, configuration, and documentation

The current synthetic smoke run proves that the software executes. It does **not** prove real-world accuracy because it uses:

- One synthetic report
- Three unverified questions
- MockLLM
- A lightweight deterministic test encoder
- No real OCR run on this machine

The next objective is to validate and improve the system on real financial reports.

---

## Non-negotiable engineering rules

1. Read the README, configuration, architecture, tests, deviations, and existing reports before modifying code.
2. Check current official documentation before using fast-changing library APIs.
3. Use pretrained encoders only. Do not train or fine-tune representation models.
4. Keep text/table embeddings separate from CLIP image embeddings.
5. Never directly compare raw similarity scores from different models or retrieval systems.
6. Fuse ranked lists using RRF or independently normalized weighted fusion.
7. Keep complete metadata on every chunk:
   - `chunk_id`
   - `doc_id`
   - page
   - modality
   - content
   - source path
   - image path
   - bounding box
   - caption
   - section heading
   - OCR status
8. Validate every generated citation against the retrieved evidence.
9. If citation validation fails, retry once with a stricter prompt. If it fails again, return `Insufficient evidence`.
10. Keep the text-only baseline fair. It may only flatten tables and remove figures.
11. Keep MockLLM and deterministic encoders for offline CI, but never silently use them in production mode.
12. Do not hard-code evaluation values or frontend results.
13. Every displayed metric must come from a saved evaluation run.
14. Chart-to-data reconstruction is out of scope.
15. Do not claim exact chart values unless they are present in caption, OCR, surrounding text, or a table.
16. Add a regression test for every bug fixed.
17. Record meaningful deviations in `docs/deviations.md`.
18. Never commit secrets or `.env` files.
19. Keep the research-prototype and not-financial-advice disclaimer.

---

# Required execution order

Complete the project in the following order:

1. Audit the current working baseline
2. Prepare the real environment and corpus
3. Improve answer quality
4. Improve search quality
5. Improve table extraction
6. Improve figure extraction
7. Improve numerical answers
8. Complete the frontend product
9. Improve evaluation
10. Complete security, reliability, and release verification

Do not make major UI changes before the backend interfaces and data models are stable.

---

# Phase 0 Audit and protect the current baseline

## Tasks

- Run all Python tests.
- Run Ruff lint and formatting checks.
- Run strict mypy checks.
- Run frontend tests.
- Build the frontend production bundle.
- Generate the synthetic fixture.
- Ingest the fixture.
- Build multimodal and baseline indexes.
- Ask one text question, one table question, and one figure question.
- Run the synthetic evaluation.
- Save the current outputs as the before-improvement baseline.
- Do not overwrite old evaluation runs.
- Confirm that production mode fails clearly when models or credentials are missing.

## Commands

```powershell
uv sync --extra dev --extra app --extra eval --extra tables
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/mmrag
npm test --prefix frontend
npm run build --prefix frontend
uv run python scripts/make_fixtures.py
uv run mmrag ingest data/raw --force
uv run mmrag index
uv run mmrag index --baseline
uv run mmrag eval
```

## Acceptance checks

- All existing tests pass.
- The frontend builds successfully.
- The fixture produces text, table, and figure chunks.
- A grounded answer contains only valid citations.
- Baseline retrieval never returns table or figure modalities.
- The current measured run is preserved.

---

# Phase 1 Prepare the real environment and corpus

## Install required components

- Install Tesseract OCR.
- Verify `tesseract --version` works in the same terminal used by the app.
- Install the `models`, `ocr`, `tables`, `app`, `eval`, and `dev` extras.
- Verify Sentence Transformers loads on CPU.
- Verify OpenCLIP loads on CPU.
- Choose a real answer-generation provider:
  - Anthropic
  - OpenAI-compatible endpoint
  - Local Ollama
- Copy `.env.example` to `.env`.
- Store API keys locally.
- Never expose keys to the frontend.
- Record exact package versions, model names, checkpoint names, and licences.

```powershell
uv sync --extra models --extra ocr --extra tables --extra app --extra eval --extra dev
tesseract --version
Copy-Item .env.example .env
```

## Collect the real corpus

- Download the 15 public annual reports listed in `data/manifests/corpus.csv`.
- Fix broken links using official investor-relations, SEC, NSE, or BSE sources.
- Keep a mixture of NSE/BSE and NASDAQ companies.
- Use FY2022-FY2024 reports.
- Include table-heavy and chart-heavy reports.
- Include at least one scanned or image-heavy report.
- Record for every report:
  - Document ID
  - Company
  - Fiscal year
  - Official URL
  - Retrieval date
  - SHA-256 checksum
  - Licence or terms
  - Page count
- Reject HTML error pages renamed as PDF.
- Skip corrupt or encrypted files with a recorded reason.
- Do not scrape protected or login-only sources.

## Inspect extraction before full indexing

Manually inspect representative pages from several reports:

- Normal paragraphs
- Multi-column pages
- Financial tables
- Tables with merged headers
- Vector charts
- Embedded images
- Scanned pages
- OCR output

Do not build the final benchmark until extraction quality is acceptable.

---

# Phase 2 Improve answer quality

## Goal

Produce clear, useful answers while remaining strictly grounded in retrieved evidence.

Better writing must never weaken citation validation or abstention.

## Provider configuration

Add configuration for:

- Provider name
- Model name
- Base URL
- Environment variable containing the key
- Timeout
- Retry count
- Maximum tokens
- Temperature
- Vision capability
- Streaming support

Add a provider health check that reports:

- Missing keys
- Unavailable Ollama model
- Endpoint connection failure
- Unsupported vision mode
- Provider timeout

Do not reveal keys in errors or logs.

## Prompt improvements

Version the generation prompt and save its hash with every answer.

The prompt must contain separate sections for:

1. User question
2. Retrieved evidence
3. Grounding rules
4. Citation rules
5. Numerical calculation rules
6. Output format

Require the LLM to:

- Use only supplied evidence
- Cite every factual claim
- State document, page, and modality
- Preserve units, currency, scale, and period
- Show calculations for numerical answers
- State assumptions
- Avoid unsupported explanations
- Abstain when evidence is insufficient

Support two answer styles:

- Compact for simple factual questions
- Detailed for comparisons, trends, and calculations

## Streaming

- Add streaming support where the provider allows it.
- Display streamed text as provisional.
- Buffer the completed answer.
- Validate citations after generation completes.
- Do not mark the answer grounded until validation succeeds.
- If validation fails, replace the provisional answer with the strict retry or `Insufficient evidence`.

## Vision

- Send only retrieved figure images to a vision-capable model.
- Include caption, OCR text, and surrounding text.
- Never send unrelated pages.
- Keep vision disabled for providers that do not support it.

## Logging

Record:

- Provider
- Model
- Prompt version and hash
- Latency
- Token usage
- Retry count
- Grounding result
- Citation-validation result
- Vision usage

## Acceptance checks

- A real provider answers text, table, figure, and multimodal questions.
- Every accepted citation resolves to a retrieved chunk.
- Invalid citations are rejected.
- An answer without citations is rejected.
- Missing credentials never trigger a silent MockLLM fallback.
- `Insufficient evidence` works correctly.
- At least 20 real answers are manually reviewed.

---

# Phase 3 Improve search quality

## Production embeddings

Use the following initial production setup:

- Text and tables: `sentence-transformers/all-MiniLM-L6-v2`
- Figures: OpenCLIP `ViT-B-32`
- Text/table index: normalized float32 vectors with exact inner-product search
- Figure index: normalized CLIP image vectors
- Figure query: CLIP text tower
- Sparse retrieval: finance-aware BM25

Persist with every index:

- Encoder ID
- Checkpoint ID
- Vector dimension
- Normalization setting
- Preprocessing version
- Corpus checksum
- Configuration hash
- Build date

Refuse to load an index created with incompatible settings.

## Embedding cache

- Cache embeddings by content hash and model identity.
- Reuse embeddings for unchanged chunks.
- Invalidate only chunks affected by document or model changes.
- Record cache hits and misses.

## Retrieval improvements

- Tune candidate K, final K, and RRF constant on a development set.
- Do not tune using final test labels.
- Benchmark:
  - Dense only
  - BM25 only
  - Hybrid RRF
  - Hybrid weighted fusion
- Add an optional cross-encoder reranker for text and serialized tables.
- Keep reranking behind a configuration flag.
- Add query normalization for:
  - Fiscal years
  - Common company abbreviations
  - Percentages
  - Currency symbols
  - Indian and international number scales
- Add conservative financial synonym expansion, for example:
  - Revenue and sales
  - Profit and earnings
  - Capex and capital expenditure
- Preserve the original query.
- Log every expansion.
- Support document, modality, and fiscal-year filters.
- Evaluate a modality-diversity rule.
- Keep diversity only if it improves verified table or figure recall without harming overall quality.

## Retrieval debugging

Return an explanation object for each result containing:

- Dense rank and score
- BM25 rank and score
- Figure rank and score
- Fusion contribution
- Final rank
- Filters applied
- Reranker score, if used

## Acceptance checks

- Text questions retrieve the correct text chunk in top five.
- Table questions retrieve the table chunk rather than only duplicate page text.
- Figure questions retrieve the linked figure using both visual and text signals.
- Save and reload produces identical deterministic rankings.
- CPU indexing and query latency are measured.
- Search improvements are supported by evaluation rather than intuition.

---

# Phase 4 Improve table extraction

## Goal

Preserve the structure that financial answers depend on:

- Table title
- Units
- Currency
- Scale
- Row labels
- Column labels
- Fiscal period
- Negative values
- Merged headers
- Multi-page continuation

## Extraction strategy

- Run PyMuPDF table detection.
- Run pdfplumber table extraction with configurable strategies.
- Compare their outputs on representative documents.
- Create a table-quality score using:
  - Non-empty cell ratio
  - Header consistency
  - Column alignment
  - Row consistency
  - Number of suspicious blank cells
  - Repeated values
  - Extraction warnings
- Select the better extractor per table or document.
- Record the selected extractor and quality score.

## Titles and units

- Search nearby text above and below the table.
- Detect table titles.
- Detect units such as:
  - USD millions
  - INR crore
  - Thousands
  - Percent
  - Basis points
- Store the source text and distance used to infer the title or unit.
- Never invent missing units.

## Complex tables

Support:

- Multi-row column headers
- Multiple row-label columns
- Merged cells
- Repeated headers
- Blank cells
- Tables continued on the next page
- Notes directly below tables

Preserve the exact display string and store a normalized numeric representation separately.

## Table serialization

Every serialized value must include explicit row and column labels.

Example:

```text
Table: Segment performance
Units: USD millions
Revenue | FY2023 = 1,100; FY2024 = 1,250
Operating profit | FY2023 = (120); FY2024 = 180
Margin | FY2023 = 10.9%; FY2024 = 14.4%
```

## Duplicate control

- Link page text and table chunks from the same region.
- Prevent duplicated table text from dominating rankings.
- Keep the original table grid for UI display.

## Debugging support

- Render page previews with table bounding boxes.
- Create a table-review view showing:
  - Original page region
  - Extracted grid
  - Serialized form
  - Units
  - Extractor
  - Quality score
  - Warnings

## Required tests

- Currency symbols
- Commas and decimals
- Percentages
- Basis points
- Negative values in parentheses
- Thousands, millions, billions, lakhs, and crores
- Merged cells
- Blank cells
- Multi-row headers
- Repeated headers
- Tables split across two pages

---

# Phase 5 Improve figure extraction

## Goal

Retrieve real charts and diagrams while rejecting logos, decorations, icons, and page scans that should not be normal figure evidence.

## Figure classification

Classify images using:

- Width and height
- Aspect ratio
- Page position
- Repetition across pages
- Color entropy
- Near-blank detection
- Caption proximity
- Surrounding keywords

Reject or heavily penalize:

- Repeated company logos
- Header and footer images
- Tiny icons
- Separators
- Decorative photos without useful context
- Near-blank images

## Scanned pages

- Treat a full-page raster as an OCR page first.
- Do not automatically create an empty figure chunk for every scanned page.
- Create a figure only if a meaningful chart or diagram region is detected.

## Vector charts

- Detect chart-like vector regions.
- Render the page.
- Crop the chart region.
- Store the crop as the figure image.
- Preserve the original page and bounding box.

## Caption and OCR linking

- Find the nearest text containing `Figure`, `Chart`, `Exhibit`, or `Source`.
- Use bounding-box distance and reading order.
- Capture surrounding paragraph text.
- Run OCR on the figure image.
- Store caption, OCR text, surrounding text, extraction method, and quality score on the same chunk.

## Figure indexing

- Index the image in the CLIP index.
- Index caption, OCR, and surrounding text in the text dense and BM25 indexes.
- Use the same `chunk_id` across those signals.

## Figure-review gallery

Add a review view showing:

- Thumbnail
- Document and page
- Bounding box
- Caption
- OCR text
- Surrounding text
- Extraction method
- Quality score
- Accepted or filtered decision

## Acceptance checks

- At least ten verified figure questions retrieve the expected chart.
- Repeated logos do not appear in normal top-K evidence.
- Figure endpoints cannot read files outside approved processed-data roots.
- Scanned pages work for OCR and page preview without becoming empty figures.
- No component claims exact chart values from visual estimation.

---

# Phase 6 Improve numerical answers

## Goal

Use deterministic code for arithmetic and unit handling instead of relying on free-form LLM calculations.

## Financial value model

Create a typed value structure containing:

- Original text
- Decimal numeric value
- Currency
- Scale
- Unit
- Percentage flag
- Fiscal period
- Row label
- Column label
- Source chunk ID
- Document and page

## Parser requirements

Support:

- Currency symbols
- Commas
- Decimals
- Percentages
- Parentheses for negatives
- Thousands
- Millions
- Billions
- Lakhs
- Crores
- Basis points

Use Python `Decimal`, not binary floating point, for financial calculations.

## Supported calculations

- Difference
- Percentage change
- Ratio
- Sum
- Average
- Minimum
- Maximum

## Calculation safety

- Require every operand to reference a retrieved table cell or exact evidence field.
- Reject calculations when periods are incompatible.
- Reject calculations when units or scales are ambiguous.
- Handle division by zero.
- Handle missing values.
- Preserve signs.
- Normalize units before arithmetic.

## Arithmetic trace

Return:

- Operands
- Formula
- Intermediate values
- Final value
- Unit
- Evidence citations

Example:

```text
Answer: Revenue increased by 13.6 percent.
Calculation: (1,250 - 1,100) / 1,100 x 100 = 13.6 percent.
Units: USD millions.
Evidence: [document-id, p.12, table]
```

The LLM may explain the verified calculation but must not replace the calculated result.

## Acceptance checks

- Exact arithmetic tests pass.
- Unit-conversion tests pass.
- Negative-value tests pass.
- Numerical tolerance is documented.
- Zero division and missing values produce safe errors or abstention.
- Every numerical result has an arithmetic trace and evidence.

---

# Phase 7 Build a complete product frontend

Keep FastAPI as the backend boundary and React with TypeScript as the primary interface. Keep Streamlit as a small compatibility client.

The main navigation should include:

- Search
- Document library
- Evaluation results
- Saved history
- Settings

## 7.1 Search workspace

Keep and improve the existing search page.

It must provide:

- A prominent question box
- Example questions
- Document scope selector
- Grounded or not-grounded badge
- Inline citation chips
- Text, table, figure, and all-evidence filters
- Real table rendering
- Figure previews
- Retrieval-score details
- Side-by-side multimodal versus baseline comparison
- Loading, empty, error, offline, and insufficient-evidence states

Clicking a citation must:

1. Highlight the citation.
2. Scroll to its evidence card.
3. Highlight the evidence card.
4. Open or focus the relevant PDF page and region when requested.

## 7.2 Document library page

List every document with:

- Filename
- Document ID
- Company
- Fiscal year
- Page count
- Processing state
- Text-chunk count
- Table count
- Figure count
- OCR-page count
- Warning count
- Last ingestion time
- Last index time

Add:

- Search
- Sorting
- Filters
- Ready, processing, failed, and needs-review status chips
- Document details
- Page thumbnails
- Table review
- Figure gallery
- Ingestion warnings
- Safe re-ingestion
- Safe re-indexing
- Removal with confirmation

Do not delete original files as a side effect of removing an index entry.

## 7.3 Upload and ingestion progress

Validate before saving:

- File extension
- MIME type
- PDF signature
- File size
- Encryption
- Duplicate checksum

Create a background job with stages:

1. Queued
2. Validating
3. Extracting text
4. Running OCR
5. Extracting tables
6. Extracting figures
7. Building embeddings
8. Updating indexes
9. Complete
10. Failed or cancelled

Expose typed job endpoints.

Use one of:

- Server-Sent Events
- WebSocket
- Polling

Show:

- Current stage
- Pages processed
- Total pages
- Elapsed time
- Warning count
- Error details
- Cancel action
- Retry action

Keep partial output isolated until processing completes. Make retries idempotent.

## 7.4 Evaluation-results page

Read only real files from `reports/<run_id>`.

Never hard-code sample metrics into the UI.

Show:

- Baseline versus multimodal results
- Recall at K
- Precision at K
- MRR
- Citation validity
- Groundedness
- Numerical exact match
- Latency
- Confidence intervals
- Sample size
- Verified-question count
- Model IDs
- Configuration
- Run time
- Git SHA

Add tabs for:

- Overall
- Question type
- Modality
- Ablations
- Failure analysis

Add a failure explorer showing:

- Question
- Reference answer
- System answer
- Retrieved evidence
- Gold evidence
- Error category
- Baseline and multimodal comparison

## 7.5 Full PDF page drawer with bounding-box highlights

Open the original PDF page from any citation or evidence card.

The drawer must support:

- Rendered page image
- Bounding-box overlay
- Correct coordinate scaling
- Page rotation
- Responsive resizing
- Zoom
- Pan
- Previous and next page
- Document and page label
- Modality and chunk ID

If a chunk has no bounding box, show the page and a clear `No region available` message.

Test overlays at multiple viewport sizes.

## 7.6 Settings for K, fusion, and providers

Expose:

- Final K
- Candidate K
- Fusion method
- RRF constant
- Dense weight
- Sparse weight
- Modality filters
- Document scope
- Provider
- Model
- Vision mode
- Demo mode

Requirements:

- Explain settings in plain language.
- Validate incompatible combinations.
- Keep API keys on the server.
- Show active model and index versions.
- Clearly show which changes require re-indexing.
- Provide reset-to-defaults.
- Save settings with history entries and evaluation requests.

## 7.7 Better mobile layout

Design for at least 360-pixel width.

Requirements:

- No full-page horizontal scrolling
- At least 44-pixel touch targets
- Visible keyboard focus
- Responsive search form
- Readable citation chips
- Collapsed advanced scores by default
- Bottom sheet or full-screen route for evidence and PDF viewing
- Responsive tables with controlled horizontal scrolling inside the table only

Test:

- Small phone
- Large phone
- Tablet
- Laptop
- Large desktop

## 7.8 Saved question history

Store:

- Question
- Answer
- Citations
- Evidence IDs
- Settings
- Document scope
- Timestamp
- Latency
- Grounding outcome
- Provider and model

Use:

- SQLite for durable local history, or
- IndexedDB only for a clearly labeled browser-local mode

Support:

- Search
- Rename
- Pin
- Rerun
- Duplicate
- Delete with confirmation
- Export as JSON
- Export as Markdown
- Clear all history with confirmation

Never store:

- API keys
- Secrets
- Raw uploaded files
- Prompts containing credentials

## 7.9 Additional product requirements

- Add side-by-side multimodal versus baseline comparison.
- Add skeleton loaders.
- Prevent duplicate submissions.
- Add request cancellation and retry.
- Add request IDs.
- Add structured backend logs.
- Add health checks and a diagnostics view.
- Add clear empty and error states.
- Add dark and light themes.
- Add keyboard navigation.
- Add ARIA labels.
- Maintain accessible contrast.
- Support reduced motion.
- Add upload-size limits.
- Sanitize filenames.
- Validate all served paths.
- Configure CORS safely.
- Do not log secrets.
- Keep the not-financial-advice disclaimer visible.

---

# Phase 8 Improve evaluation

Evaluation must prove whether changes improve the system.

Do not tune on the final test questions.

## Gold question set

Create at least 60 questions over the real corpus, balanced across:

- Text
- Table
- Figure
- Multimodal

Each record must include:

- Question ID
- Question
- Reference answer
- Gold document
- Gold page
- Gold modality
- Gold chunk IDs
- Units, if applicable
- Question type
- Author
- Reviewer
- Verified flag

A human reviewer must open the cited page and confirm:

- The question is unambiguous.
- The reference answer is correct.
- Units and periods are correct.
- All supporting evidence is included.
- The question type is correct.

Keep separate:

- Development split for tuning
- Locked test split for final reporting

## External benchmarks

- Load official FinQA contexts and supporting labels.
- Load official TAT-QA contexts and labels.
- Use a small seeded DocVQA subset.
- Use a small seeded SlideVQA subset.
- Record versions and licences.
- Do not imply that FinQA or TAT-QA measures figure retrieval.

## Retrieval metrics

Report:

- Recall at 1, 3, 5, and 10
- Precision at 1, 3, 5, and 10
- MRR

Report overall and by:

- Question type
- Modality
- Document
- Verified subset

## Answer metrics

Report:

- Citation validity rate
- Groundedness
- Answer relevance
- Numerical exact match
- Official benchmark answer metric, where applicable
- Latency
- Token usage

If using an LLM judge:

- Use a fixed versioned rubric.
- Use temperature zero.
- Prefer a different judge model from the generator.
- Manually inspect at least 20 judge outputs.
- Report agreement with human review.

## Required comparisons

Run all comparisons on identical questions:

- Text-only baseline versus multimodal
- Dense only versus BM25 only versus hybrid
- RRF versus weighted fusion
- Text only versus text plus tables versus text plus tables plus figures
- Without versus with reranker
- Encoder alternatives, if tested

## Statistics

- Use fixed seeds.
- Report bootstrap 95% confidence intervals.
- Use a paired significance test for baseline versus multimodal.
- State sample sizes.
- Do not overclaim non-significant changes.

## Failure analysis categories

Classify failures as:

- Extraction error
- Correct evidence ranked too low
- Wrong modality
- Correct evidence but wrong answer
- Arithmetic error
- Unit or scale error
- Invalid citation
- Unsupported answer
- Ambiguous question
- Incomplete gold label
- Model or provider failure
- Out-of-scope visual interpretation

## Evaluation outputs

Every run must create:

```text
reports/<run_id>/
  config.yaml or config.json
  git_sha
  environment.json
  per_question.jsonl
  metrics.json
  summary.md
  plots/
```

Add regression thresholds so CI warns when verified fixture or gold metrics drop beyond a documented tolerance.

---

# Phase 9 Testing, security, and reliability

## Backend tests

Add tests for:

- Financial value parsing
- Unit normalization
- Deterministic calculations
- Table-quality scoring
- Figure filtering
- Query normalization
- Fusion
- Reranking
- Prompt rendering
- Citation validation
- Upload validation
- Job-state transitions
- Settings validation
- History storage
- Safe path handling

## Integration tests

Cover PDFs containing:

- Digital text
- Scanned pages
- Multi-column text
- Multi-page tables
- Merged table headers
- Embedded charts
- Vector charts
- Repeated logos
- Corrupt or encrypted inputs

## API tests

Cover:

- Uploads
- Progress jobs
- Documents
- Re-ingestion
- Retrieval
- Answers
- Figure serving
- Page rendering
- Settings
- History
- Evaluation reports
- Error states

## End-to-end tests

With MockLLM, test:

- Grounded answer
- Insufficient evidence
- Invalid citation rejection
- Text question
- Table question
- Figure question
- Multimodal question
- Baseline mode
- Multimodal mode

## Frontend tests

Add component tests for:

- Citation chips
- Evidence tabs
- Tables
- Figures
- Progress states
- Settings
- History
- Evaluation charts
- PDF overlays
- Error and empty states

Add Playwright flows for:

- Text, table, and figure questions
- Document upload
- Ingestion progress
- Document library
- Compare mode
- PDF source drawer
- Evaluation page
- Settings changes
- Dark mode
- Mobile layout
- Saved history

Fail browser tests on:

- Console errors
- Failed API requests
- Unhandled exceptions
- Broken images
- Inaccessible controls

Save and manually review final screenshots for:

- Light desktop
- Dark desktop
- Mobile
- Empty state
- Loading state
- Insufficient evidence
- Upload progress
- Evaluation results

## Coverage

- Keep offline CI network-free and GPU-free.
- Keep core line coverage at or above 80%.
- Add a regression test for every fixed bug.

## Security

- Never expose API keys to the browser.
- Never log secrets.
- Validate all file paths against approved roots.
- Sanitize filenames.
- Enforce file-size and page-count limits.
- Reject non-PDF uploads.
- Use parameterized SQLite queries.
- Validate requests and responses with typed schemas.
- Configure CORS for intended origins.
- Document whether evidence is sent to an external provider.
- Provide a local-only Ollama option.

---

# Phase 10 Final verification and handoff

## Required final checks

- Verify setup in a clean environment.
- Run fixture ingestion and indexing.
- Run real-corpus ingestion and indexing.
- Run baseline and multimodal evaluation.
- Run API tests.
- Run React tests and build.
- Run Streamlit smoke test.
- Run Docker build and startup.
- Run Playwright E2E.
- Measure ingestion time.
- Measure indexing time.
- Measure per-query latency.
- Record test-machine information.

## Documentation updates

Update:

- `README.md`
- `docs/architecture.md`
- `docs/evaluation.md`
- `docs/results.md`
- `docs/deviations.md`
- `docs/gold_set_review.md`
- Relevant ADRs
- Colab quick-start notebook
- Docker instructions

Update the academic report and presentation only with measured run IDs and real screenshots.

## Final handoff must contain

1. What was implemented
2. Which files changed
3. Test and build results
4. Headline baseline-versus-multimodal metrics with run ID
5. Exact setup and run commands
6. Known limitations
7. Manual downloads still required
8. Human gold-question verification still required
9. Credentials or local models still required
10. Licence and privacy notes

---

# Commands expected in the finished project

```powershell
# Install
uv sync --extra models --extra ocr --extra tables --extra app --extra eval --extra dev
npm install --prefix frontend

# Prepare data
uv run python scripts/download_corpus.py
uv run mmrag ingest data/raw --force

# Build indexes
uv run mmrag index
uv run mmrag index --baseline

# Ask questions
uv run mmrag ask "What was revenue in FY2024?"

# Evaluate
uv run mmrag eval --gold data/eval/own_corpus_gold.jsonl

# Run the product
make app

# Run Streamlit
make streamlit

# Verify Python
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/mmrag

# Verify frontend
npm test --prefix frontend
npm run build --prefix frontend
```

---

# Definition of done

The project is complete only when all of the following are true:

- [ ] Fifteen public reports are present or have documented manual-download actions.
- [ ] Every report is traceable through the manifest.
- [ ] OCR works on a real scanned page.
- [ ] Text, table, and figure extraction are manually inspected.
- [ ] Production encoders build and reload versioned indexes.
- [ ] A real API or local LLM produces citation-validated answers.
- [ ] Unsupported answers are rejected.
- [ ] Numerical answers use deterministic arithmetic.
- [ ] Numerical answers show formulas, units, and evidence.
- [ ] The document library works.
- [ ] Upload and ingestion progress works.
- [ ] The evaluation-results page uses real run artifacts.
- [ ] The PDF drawer highlights evidence bounding boxes.
- [ ] Retrieval and provider settings work.
- [ ] The app works at 360-pixel mobile width.
- [ ] Saved question history works.
- [ ] React compare mode works.
- [ ] The baseline and multimodal systems use identical questions.
- [ ] At least 60 gold questions exist.
- [ ] Headline results use human-verified questions.
- [ ] Confidence intervals, ablations, and failure analysis are reported.
- [ ] Python tests, lint, formatting, and typing pass.
- [ ] Frontend tests and build pass.
- [ ] Browser E2E passes without console errors.
- [ ] Docker setup works.
- [ ] No secrets are committed.
- [ ] No fake metrics or placeholder results remain.
- [ ] No unfinished `TODO` or `pass` remains in delivered code paths.
- [ ] Every deviation is documented.

---

# Agent progress-report format

At every milestone, report only:

1. What was implemented
2. Which files changed
3. Which commands were run
4. Whether each command passed or failed
5. Measured acceptance evidence
6. Known limitation or next blocker

Do not claim success without executed evidence.

If real documents, model weights, API keys, or human verification are unavailable:

- Complete all possible offline work.
- Do not fabricate the missing result.
- Record the exact missing input.
- Leave one clear command or action for a human to continue.

Begin with Phase 0 and proceed phase by phase until the definition of done is satisfied.
