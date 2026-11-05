"""Evaluation runner that writes provenance-complete, measured report artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from mmrag.evaluation.metrics import (
    bootstrap_mean_ci,
    numeric_exact_match,
    paired_permutation_test,
    retrieval_metrics,
)
from mmrag.schemas import Modality


class EvaluationCase(BaseModel):
    question_id: str
    question: str
    relevant_chunk_ids: set[str]
    reference_answer: str
    question_type: str
    verified: bool = False
    reference_document: str | None = None
    reference_pages: list[int] = []
    reference_modalities: list[Modality] = []
    units: str | None = None
    fiscal_period: str | None = None
    author: str | None = None
    reviewer: str | None = None


class SystemOutput(BaseModel):
    ranked_chunk_ids: list[str]
    answer: str
    grounded: bool
    citation_valid: bool
    latency_ms: float = 0
    token_usage: int | None = None


SystemCallable = Callable[[EvaluationCase], SystemOutput]


class PerQuestionResult(BaseModel):
    question_id: str
    question_type: str
    system: str
    recall: dict[int, float]
    precision: dict[int, float]
    mrr: float
    numeric_exact_match: float
    grounded: bool
    citation_valid: bool
    answer: str
    ranked_chunk_ids: list[str]
    latency_ms: float
    token_usage: int | None
    failure_categories: list[str]
    verified: bool
    reference_document: str | None
    reference_modalities: list[Modality]


class EvaluationRunner:
    def __init__(self, reports_dir: Path, *, seed: int = 42) -> None:
        self.reports_dir = reports_dir
        self.seed = seed

    @staticmethod
    def _git_sha() -> str:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    def run(
        self,
        cases: list[EvaluationCase],
        systems: dict[str, SystemCallable],
        *,
        config: dict[str, object],
        run_id: str | None = None,
    ) -> Path:
        if not cases or not systems:
            raise ValueError("Evaluation requires at least one case and one system")
        run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = self.reports_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=False)

        results: list[PerQuestionResult] = []
        by_system: dict[str, list[PerQuestionResult]] = {name: [] for name in systems}
        for case in cases:
            for name, system in systems.items():
                output = system(case)
                metrics = retrieval_metrics(output.ranked_chunk_ids, case.relevant_chunk_ids)
                result = PerQuestionResult(
                    question_id=case.question_id,
                    question_type=case.question_type,
                    system=name,
                    recall=metrics.recall,
                    precision=metrics.precision,
                    mrr=metrics.mrr,
                    numeric_exact_match=numeric_exact_match(output.answer, case.reference_answer),
                    grounded=output.grounded,
                    citation_valid=output.citation_valid,
                    answer=output.answer,
                    ranked_chunk_ids=output.ranked_chunk_ids,
                    latency_ms=output.latency_ms,
                    token_usage=output.token_usage,
                    failure_categories=self._failure_categories(
                        metrics.recall[10], output, numeric_exact_match(output.answer, case.reference_answer)
                    ),
                    verified=case.verified,
                    reference_document=case.reference_document,
                    reference_modalities=case.reference_modalities,
                )
                results.append(result)
                by_system[name].append(result)

        metrics_payload: dict[str, object] = {
            "run_id": run_id,
            "sample_size": len(cases),
            "verified_sample_size": sum(case.verified for case in cases),
            "systems": {},
        }
        system_metrics: dict[str, object] = {}
        for name, system_results in by_system.items():
            summary: dict[str, object] = {}
            for field in ("mrr", "numeric_exact_match"):
                values = [float(getattr(result, field)) for result in system_results]
                mean, low, high = bootstrap_mean_ci(values, seed=self.seed)
                summary[field] = {"mean": mean, "ci95": [low, high]}
            for k in (1, 3, 5, 10):
                for metric_name in ("recall", "precision"):
                    values = [getattr(result, metric_name)[k] for result in system_results]
                    mean, low, high = bootstrap_mean_ci(values, seed=self.seed)
                    summary[f"{metric_name}@{k}"] = {"mean": mean, "ci95": [low, high]}
            summary["citation_valid_rate"] = sum(r.citation_valid for r in system_results) / len(system_results)
            summary["grounded_rate"] = sum(r.grounded for r in system_results) / len(system_results)
            latency = [result.latency_ms for result in system_results]
            mean, low, high = bootstrap_mean_ci(latency, seed=self.seed)
            summary["latency_ms"] = {"mean": mean, "ci95": [low, high]}
            tokens = [float(result.token_usage) for result in system_results if result.token_usage is not None]
            summary["token_usage"] = (
                {"mean": sum(tokens) / len(tokens), "sample_size": len(tokens)} if tokens else {"sample_size": 0}
            )
            categories: dict[str, int] = {}
            for result in system_results:
                for category in result.failure_categories:
                    categories[category] = categories.get(category, 0) + 1
            summary["failure_categories"] = categories
            summary["breakdowns"] = self._breakdowns(system_results)
            system_metrics[name] = summary
        metrics_payload["systems"] = system_metrics

        if "baseline" in by_system and "multimodal" in by_system:
            base = [result.mrr for result in by_system["baseline"]]
            multi = [result.mrr for result in by_system["multimodal"]]
            metrics_payload["baseline_vs_multimodal_mrr_p_value"] = paired_permutation_test(base, multi, seed=self.seed)

        (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "git_sha").write_text(self._git_sha() + "\n", encoding="utf-8")
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        }
        (output_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
        )
        with (output_dir / "per_question.jsonl").open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(result.model_dump_json() + "\n")
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._write_summary(output_dir / "summary.md", metrics_payload)
        self._write_plot(output_dir / "plots" / "recall_at_5.svg", system_metrics)
        return output_dir

    @staticmethod
    def _failure_categories(recall_at_10: float, output: SystemOutput, numeric_match: float) -> list[str]:
        failures: list[str] = []
        if recall_at_10 == 0:
            failures.append("retrieval_miss")
        if not output.citation_valid:
            failures.append("citation_invalid")
        if not output.grounded:
            failures.append("ungrounded")
        if numeric_match == 0:
            failures.append("numeric_mismatch")
        return failures

    @staticmethod
    def _breakdowns(results: list[PerQuestionResult]) -> dict[str, dict[str, dict[str, float]]]:
        groups: dict[str, dict[str, list[PerQuestionResult]]] = {
            "question_type": {},
            "modality": {},
            "document": {},
            "verified": {},
        }
        for result in results:
            groups["question_type"].setdefault(result.question_type, []).append(result)
            for modality in result.reference_modalities:
                groups["modality"].setdefault(modality.value, []).append(result)
            if result.reference_document:
                groups["document"].setdefault(result.reference_document, []).append(result)
            groups["verified"].setdefault(str(result.verified).lower(), []).append(result)
        return {
            dimension: {
                name: {
                    "sample_size": float(len(items)),
                    "mrr": sum(item.mrr for item in items) / len(items),
                    "recall@5": sum(item.recall[5] for item in items) / len(items),
                }
                for name, items in values.items()
            }
            for dimension, values in groups.items()
        }

    @staticmethod
    def _write_plot(path: Path, systems: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        bars: list[str] = []
        labels: list[str] = []
        for index, (name, raw) in enumerate(systems.items()):
            assert isinstance(raw, dict)
            metric = raw["recall@5"]
            assert isinstance(metric, dict)
            value = float(metric["mean"])
            height = value * 160
            x = 50 + index * 120
            bars.append(f'<rect x="{x}" y="{190 - height:.1f}" width="70" height="{height:.1f}" fill="#24704f"/>')
            labels.append(f'<text x="{x}" y="215" font-size="12">{name}</text>')
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="500" height="240" role="img" '
            'aria-label="Recall at 5 by system"><line x1="35" y1="30" x2="35" y2="190" stroke="#222"/>'
            + "".join(bars + labels)
            + "</svg>"
        )
        path.write_text(svg, encoding="utf-8")

    @staticmethod
    def _write_summary(path: Path, metrics: dict[str, object]) -> None:
        lines = [
            f"# Evaluation run {metrics['run_id']}",
            "",
            f"Measured questions: {metrics['sample_size']} ({metrics['verified_sample_size']} human-verified).",
            "",
            "| System | MRR (95% CI) | Recall@5 (95% CI) | Numeric EM (95% CI) |",
            "|---|---:|---:|---:|",
        ]
        systems = metrics["systems"]
        assert isinstance(systems, dict)
        for name, raw_summary in systems.items():
            assert isinstance(raw_summary, dict)
            formatted: list[str] = []
            for key in ("mrr", "recall@5", "numeric_exact_match"):
                value = raw_summary[key]
                assert isinstance(value, dict)
                ci = value["ci95"]
                assert isinstance(ci, list)
                formatted.append(f"{value['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
            lines.append(f"| {name} | {' | '.join(formatted)} |")
        lines.extend(["", "All figures above are computed from `per_question.jsonl` in this run."])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def regression_failures(
    current: dict[str, float], reference: dict[str, float], tolerances: dict[str, float]
) -> list[str]:
    """Return metric regressions beyond documented tolerances."""
    return [
        metric
        for metric, reference_value in reference.items()
        if metric in current and current[metric] < reference_value - tolerances.get(metric, 0.0)
    ]
