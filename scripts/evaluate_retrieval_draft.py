"""Evaluate retrieval on the source-grounded AI draft without calling an answer LLM."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mmrag.evaluation.metrics import bootstrap_mean_ci, paired_permutation_test, retrieval_metrics
from mmrag.evaluation.runner import EvaluationCase
from mmrag.pipeline import MMRAGPipeline


def _mean_ci(values: list[float], seed: int) -> dict[str, object]:
    mean, low, high = bootstrap_mean_ci(values, seed=seed)
    return {"mean": mean, "ci95": [low, high]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("data/eval/own_corpus_gold.ai_draft.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/production.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/real/retrieval_ai_draft"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    cases = [
        EvaluationCase.model_validate_json(line)
        for line in args.gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pipeline = MMRAGPipeline.from_config(args.config)
    per_question: list[dict[str, Any]] = []
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        for system, baseline in (("baseline", True), ("multimodal", False)):
            result = pipeline.retrieve(case.question, k=args.k, baseline=baseline)
            ranked = [item.chunk.chunk_id for item in result.evidence]
            scores = retrieval_metrics(ranked, case.relevant_chunk_ids)
            row = {
                "question_id": case.question_id,
                "question_type": case.question_type,
                "reference_document": case.reference_document,
                "reference_modalities": [item.value for item in case.reference_modalities],
                "verified": case.verified,
                "system": system,
                "ranked_chunk_ids": ranked,
                "relevant_chunk_ids": sorted(case.relevant_chunk_ids),
                "recall": scores.recall,
                "precision": scores.precision,
                "mrr": scores.mrr,
                "latency_ms": result.latency_ms,
                "retrieval_miss_at_10": scores.recall[10] == 0,
            }
            per_question.append(row)
            by_system[system].append(row)

    systems: dict[str, object] = {}
    for name, rows in by_system.items():
        summary: dict[str, object] = {
            "mrr": _mean_ci([float(row["mrr"]) for row in rows], pipeline.config.seed),
            "latency_ms": _mean_ci([float(row["latency_ms"]) for row in rows], pipeline.config.seed),
            "retrieval_misses_at_10": sum(bool(row["retrieval_miss_at_10"]) for row in rows),
        }
        for k in (1, 3, 5, 10):
            for metric in ("recall", "precision"):
                values = [float(row[metric][k]) for row in rows]
                summary[f"{metric}@{k}"] = _mean_ci(values, pipeline.config.seed)
        breakdowns: dict[str, object] = {}
        for question_type in sorted({str(row["question_type"]) for row in rows}):
            selected = [row for row in rows if row["question_type"] == question_type]
            breakdowns[question_type] = {
                "sample_size": len(selected),
                "mrr": sum(float(row["mrr"]) for row in selected) / len(selected),
                "recall@5": sum(float(row["recall"][5]) for row in selected) / len(selected),
            }
        summary["by_question_type"] = breakdowns
        systems[name] = summary

    baseline_mrr = [float(row["mrr"]) for row in by_system["baseline"]]
    multimodal_mrr = [float(row["mrr"]) for row in by_system["multimodal"]]
    verified_sample_size = sum(case.verified for case in cases)
    dataset_status = (
        "Human-verified source-grounded evaluation set"
        if verified_sample_size == len(cases)
        else "Source-grounded draft; independent human review pending"
    )
    payload = {
        "evaluation_mode": "retrieval_only",
        "dataset_status": dataset_status,
        "sample_size": len(cases),
        "verified_sample_size": verified_sample_size,
        "top_k": args.k,
        "systems": systems,
        "baseline_vs_multimodal_mrr_p_value": paired_permutation_test(
            baseline_mrr, multimodal_mrr, seed=pipeline.config.seed
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output / "per_question.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_question:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = [
        "# Real-corpus retrieval evaluation",
        "",
        f"This is a retrieval-only evaluation over {verified_sample_size} human-verified source-grounded questions."
        if verified_sample_size == len(cases)
        else "This is a retrieval-only evaluation over a draft question set; independent human review remains pending.",
        "",
        "| System | MRR | Recall@5 | Recall@10 | Mean latency (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, raw in systems.items():
        assert isinstance(raw, dict)
        lines.append(
            f"| {name} | {raw['mrr']['mean']:.3f} | {raw['recall@5']['mean']:.3f} | "
            f"{raw['recall@10']['mean']:.3f} | {raw['latency_ms']['mean']:.1f} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"questions={len(cases)} systems=2 output={args.output}")


if __name__ == "__main__":
    main()
