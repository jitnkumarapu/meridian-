import json

import pytest

from mmrag.evaluation import (
    EvaluationCase,
    EvaluationRunner,
    SystemOutput,
    bootstrap_mean_ci,
    numeric_exact_match,
    paired_permutation_test,
    retrieval_metrics,
)


def test_retrieval_metrics_are_hand_computed() -> None:
    result = retrieval_metrics(["x", "b", "a"], {"a", "b"}, ks=(1, 3))
    assert result.recall == {1: 0.0, 3: 1.0}
    assert result.precision == {1: 0.0, 3: pytest.approx(2 / 3)}
    assert result.mrr == 0.5


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [("₹1,200.00 crore", "1200", 1.0), ("(25)", "-25", 1.0), ("12", "13", 0.0)],
)
def test_numeric_exact_match(prediction: str, reference: str, expected: float) -> None:
    assert numeric_exact_match(prediction, reference) == expected


def test_statistics_are_deterministic() -> None:
    assert bootstrap_mean_ci([0.0, 1.0, 1.0], samples=100, seed=7) == bootstrap_mean_ci(
        [0.0, 1.0, 1.0], samples=100, seed=7
    )
    assert paired_permutation_test([0, 0], [1, 1]) == 0.5
    assert paired_permutation_test([1, 1], [1, 1]) == 1.0


def test_runner_writes_measured_artifacts(tmp_path) -> None:
    case = EvaluationCase(
        question_id="q1",
        question="Revenue?",
        relevant_chunk_ids={"table"},
        reference_answer="120",
        question_type="table",
        verified=True,
    )

    def baseline(_: EvaluationCase) -> SystemOutput:
        return SystemOutput(ranked_chunk_ids=["text"], answer="100", grounded=True, citation_valid=True)

    def multimodal(_: EvaluationCase) -> SystemOutput:
        return SystemOutput(ranked_chunk_ids=["table"], answer="120", grounded=True, citation_valid=True)

    output = EvaluationRunner(tmp_path).run(
        [case], {"baseline": baseline, "multimodal": multimodal}, config={"seed": 42}, run_id="test"
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["systems"]["multimodal"]["mrr"]["mean"] == 1.0
    assert metrics["systems"]["multimodal"]["precision@1"]["mean"] == 1.0
    assert metrics["systems"]["multimodal"]["token_usage"]["sample_size"] == 0
    assert (output / "per_question.jsonl").exists()
    assert (output / "environment.json").exists()
    assert (output / "plots" / "recall_at_5.svg").exists()
    assert "Measured questions: 1" in (output / "summary.md").read_text(encoding="utf-8")


def test_regression_thresholds_are_explicit() -> None:
    from mmrag.evaluation import regression_failures

    assert regression_failures({"mrr": 0.7}, {"mrr": 0.8}, {"mrr": 0.05}) == ["mrr"]
    assert regression_failures({"mrr": 0.76}, {"mrr": 0.8}, {"mrr": 0.05}) == []
