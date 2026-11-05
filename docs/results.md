# Results

## Real corpus

The extraction audit passed for all 15 reports:

- 3,868 PDF pages
- 14,145 chunks
- 15/15 reports reviewed
- No document failures

The machine-readable details are in
`reports/real/extraction_review.json`, with a readable table in
`reports/real/extraction_review.md`.

The 60-question corpus file is an independently reviewable draft, not benchmark truth. Its records
are intentionally left unverified until a different human checks the original PDFs.

## Synthetic smoke evaluation

The saved fixture runs demonstrate deterministic plumbing and citation validation only. They use
three synthetic questions and must not be interpreted as real-corpus quality measurements. See the
run directories under `reports/` for the exact configuration and per-question records.

