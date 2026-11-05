# Architecture and objective traceability

The invariant is traceability: every candidate remains a `Chunk` with document, one-based page,
modality, source path, and optional bounding box. Dense and sparse components return ranks for those
IDs; only fusion creates a final order. Generation cannot invent a new source because the citation
gate resolves labels back to the retrieved chunk set.

| Objective | Implementation | Verification |
|---|---|---|
| O1 PDF/OCR extraction | `ingestion/`, `extraction/` | fixture ingestion and OCR-backend tests |
| O2 pretrained embeddings | `embedding/` | adapter and normalisation tests; explicit test encoder |
| O3 hybrid retrieval | `indexing/`, `retrieval/` | persistence, tokenizer, exact fusion tests |
| O4 grounded answers | `generation/` | parser, validator, retry/abstain E2E tests |
| O5 fair baseline | `baseline/`, `evaluation/` | identical-query comparison and metric tests |
| O6 generalisation | benchmark download/load scripts | seeded-subset manifest and evaluation output |
| O7 interactive demo | `api/`, `frontend/`, `app/` | API tests and browser smoke workflow |

The deterministic hash encoder and MockLLM exist only for offline CI. Production configurations
select pretrained encoders and an explicit generation provider; missing dependencies or credentials
raise actionable errors rather than silently changing the experiment.

Index directories include encoder identity, dimensions, normalization, preprocessing version, corpus
checksum, fusion configuration, and build time. Retrieval responses retain the original query plus
normalization expansions and per-source ranks/scores/contributions. Financial calculations accept only
values carrying chunk/document/page provenance and use `Decimal` throughout.
