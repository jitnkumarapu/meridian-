# Changelog

## 2026-11-05

### Changed

- Reframed the frontend navigation around Ask, Documents, Quality report, and Settings.
- Reworked the Ask page with document scope, staged loading/cancel feedback, recent questions, grounded-answer status, citations, and evidence panels.
- Redesigned Documents with drag-and-drop upload, supported-file validation, processing progress, extraction counts, previews, removal, and per-document search inclusion toggles.
- Replaced raw evaluation JSON as the primary view with plain-language quality metrics while retaining detailed results behind an expandable section.
- Simplified Settings into answer detail and source-count controls, with developer options kept under Advanced and visible save/reset feedback.
- Added an initial codebase audit and ordered implementation plan in [AUDIT.md](./AUDIT.md).

### Known limitations

- The API does not yet expose a persisted document-inclusion setting; the frontend persists inclusion choices in browser local storage and sends the selected document IDs with each Ask request.
- Quality report re-running and API-key connection testing are not available in the current backend contract.
- Upload processing remains server-side and reports progress from the existing job polling endpoint.

