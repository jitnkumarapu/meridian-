# Evaluation protocol

All comparisons consume the same ordered questions, K values, generator prompt, and seed. Baseline
changes only two corpus transforms: it serialises tables as plain row text and excludes figures.
Retrieval reports Recall@K, Precision@K, and MRR overall and by question type. Answer checks include
citation validity and numeric exact match. Bootstrap confidence intervals resample questions with a
fixed seed; paired comparison operates on per-question deltas.

Each run writes its full configuration, raw per-question records, metrics, summary, and plots under
`reports/<run_id>/`. Synthetic fixture runs are not evidence of real-corpus quality and are labelled.
Unverified machine-drafted gold questions are excluded from the verified-subset headline.

Required real-data ablations are: baseline vs multimodal, dense vs sparse vs hybrid, RRF vs weighted
sum, and text → tables → figures. The full-corpus run remains pending until the public PDFs and human
verified gold set are present.
