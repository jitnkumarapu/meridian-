# Deviations and environment constraints

- Tesseract was not available on `PATH` during the 2026-11-04 final verification. OCR code supports
  it and tests use an injected deterministic OCR backend; a real scanned-report OCR review remains pending.
- The Phase 1 pilot counts are reference values only. No source pilot PDFs were supplied, so no count
  comparison can be made yet.
- Git metadata is mounted read-only in the managed workspace, so milestone commits may be unavailable.
- Real corpus downloads, benchmark downloads, pretrained-model runs, and API-LLM runs are reported
  only when actually executed; synthetic fixture results are labelled as such.
- Docker was not available on `PATH` during the 2026-11-04 verification, so a clean-container build
  could not be executed on this host.
- Playwright Chromium automation is now verified on desktop and 360px mobile. The tested product flow
  has no axe violations or page-level horizontal overflow; screenshots and traces are retained on failure.
