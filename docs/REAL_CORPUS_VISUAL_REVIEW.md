# Real-corpus visual source review

Review date: 2026-11-05

This review uses rendered source pages, not extracted Markdown alone. The question draft remains
`verified: false` until a separate human reviewer checks every answer and signs it off.

## Reviewed visual evidence

| Document | PDF page(s) | Visual finding used for the question draft |
|---|---:|---|
| Infosys FY2024 | 15, 31 | Business KPI dashboard and geographic revenue-distribution graphic are legible; units and fiscal-year labels are explicit. |
| TCS FY2024 | 17 | Five-year revenue, operating-profit, EPS, cash-conversion, and payout charts are legible and labelled in ₹ crore/percent. |
| Wipro FY2024 | 23, 37 | Financial table is legible; value-creation model clearly separates inputs, value created, and stakeholder outcomes. |
| HCLTech FY2024 | 15 | Five-year KPI bars, employee headcount, segment-revenue donut, and client-category trends are legible. |
| Reliance FY2024 | 3 | Financial-metrics dashboard clearly distinguishes revenue, PAT, EPS, balance-sheet metrics, and business summaries. |
| HDFC Bank FY2024 | 8 | Stakeholder highlights clearly show branch, advances, customer, employee, and learning metrics. |
| ITC FY2024 | 5 | Triple-bottom-line infographic clearly separates economic, environmental, and social outcomes. |
| Asian Paints FY2024 | 20, 81 | ESG value-creation diagram is legible; the ten-year review is dense and should be consumed primarily as a table. |
| NVIDIA FY2024 | 20 | Business overview clearly shows consolidated results, reportable segments, and four market-platform outcomes. |
| Intel FY2023 | 40 | Segment-revenue waterfall clearly reconciles the 2022 starting value to the 2023 ending value. |
| Cisco FY2024 | 6 | Revenue mix, three-year revenue trend, operating cash flow, and margin charts are legible; rounded components are noted. |

## Plain-report visual checks

Apple, Microsoft, Alphabet, and Amazon use predominantly prose and financial tables in the reviewed
pages rather than meaningful charts. Their draft questions therefore use text, table, and
multimodal evidence instead of inventing figure requirements. Alphabet pages 4-5, Amazon page 3,
Apple page 3, and Microsoft page 6 rendered without clipping or broken glyphs.

## Review decisions

- Figure questions are limited to pages where the layout or diagram genuinely carries information.
- Dense financial-statement pages use table evidence, even when visually formatted.
- Rounded dashboard values are labelled as rounded in reference answers when they do not sum to the
  audited statement total.
- Page numbers are PDF page indices used by the ingestion pipeline, not printed footer numbers.
- Extraction counts and warnings are audited separately in `reports/real/extraction_review.md`.
