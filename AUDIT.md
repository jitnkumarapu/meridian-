# Meridian initial audit

Date: 2026-11-05

## What I inspected

- Started the existing Vite frontend at `http://127.0.0.1:5175/`.
- Clicked through Search, Library, Evaluations, History, and Settings.
- Reviewed the current React components, styles, API surface, and available fixture data.

## Findings

### Navigation and information architecture

- The primary navigation is labelled `Search`, `Library`, `Evaluations`, `History`, and `Settings`, which does not match the requested user-facing model of Ask, Documents, Settings, and Quality report.
- History is a useful capability but is currently a separate top-level destination competing with the core workflow instead of being a compact recent-question affordance on Ask.
- The footer still presents the product as a research prototype.

### Ask/search experience

- The Ask surface is visually strong but still uses prototype language and exposes retrieval controls such as modality and baseline comparison.
- The answer area needs a single coherent response with explicit short answer, explanation, evidence, sources, and follow-ups.
- Loading is represented as a generic skeleton; it does not communicate the requested retrieval stages.
- Scope selection exists in code but is not clearly framed as “All documents” versus a specific document.
- Existing answer rendering supports paragraphs and bullets, but not structured headings, inline numbered citations, or follow-up actions.

### Documents

- The current page is titled “Document library” and the upload action is “Upload PDF”; the purpose and supported file handling are not explained.
- Document rows show extraction counts, but do not expose an include/exclude toggle, date added, Preview/Remove actions, or a clear empty state.
- Upload errors are surfaced as raw exception text and upload progress is only represented by polling job records.
- There is no visible explanation for failed or needs-review documents.

### Quality report

- The evaluation page renders large raw JSON blobs in scrollable cards, including internal metric names and cryptic values.
- It has no plain-language metric explanations, visual summary, citation accuracy/refusal reporting, or re-run control.
- The current fixture demonstrates weak retrieval recall, especially for figures and tables; this must be reported honestly rather than hidden.

### Settings

- Settings expose internal retrieval knobs (K values, fusion constants, dense/sparse weights, provider/model) without plain-language descriptions.
- There is no API-key/connection test affordance, reset feedback, or clear separation between user settings and developer-only controls.
- The settings page does not clearly communicate which options affect answer behavior.

### Reliability and completeness

- The frontend has a broad API integration but limited user-facing error classification and no explicit handling for empty/overlong queries.
- The upload request hard-codes `application/pdf`, so image/scanned upload support is not represented in the client.
- The current UI does not visibly verify that answer numbers are present in retrieved evidence.
- The current repo contains evaluation fixtures and tests, but no user-facing golden-set/quality report workflow in the frontend.

## Ordered implementation plan

1. Establish the final navigation and terminology: Ask, Documents, Settings, Quality report; keep recent history accessible from Ask.
2. Improve the Ask workflow and answer renderer: scope selector, staged loading, structured answer sections, citations, evidence, and follow-ups.
3. Redesign Documents with clear status, inclusion toggles, preview/remove actions, upload drop zone, progress, and actionable errors.
4. Replace raw evaluation JSON with a plain-language Quality report and re-run action.
5. Simplify Settings into user-facing controls, move developer knobs under Advanced, and add validation/feedback.
6. Add responsive/accessibility polish and remove prototype/debug language.
7. Update tests and documentation, then run targeted frontend tests/build and exercise the browser flows again.

