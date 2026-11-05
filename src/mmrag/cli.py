"""Command-line interface for the reproducible pipeline."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from mmrag.evaluation import EvaluationCase, EvaluationRunner, SystemOutput
from mmrag.logging import configure_logging
from mmrag.pipeline import MMRAGPipeline
from mmrag.schemas import Modality

app = typer.Typer(no_args_is_help=True, help="Multimodal financial-document retrieval")
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


def _pipeline(config: Path) -> MMRAGPipeline:
    configure_logging()
    return MMRAGPipeline.from_config(config)


@app.command()
def ingest(
    source: Annotated[Path, typer.Argument(exists=True)],
    config: ConfigOption = Path("configs/default.yaml"),
    force: Annotated[bool, typer.Option("--force")] = False,
    manifest: Annotated[Path | None, typer.Option("--manifest", exists=True, dir_okay=False)] = None,
) -> None:
    """Extract PDFs from SOURCE, optionally restricted to a corpus manifest."""
    pipeline = _pipeline(config)
    if manifest is None:
        reports = pipeline.ingest(source, force=force)
    else:
        if not source.is_dir():
            raise typer.BadParameter("SOURCE must be a directory when --manifest is used")
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        filenames = [row.get("filename", "").strip() for row in rows if row.get("status") == "ready"]
        if not filenames:
            raise typer.BadParameter("manifest contains no ready documents")
        reports = []
        for filename in filenames:
            if not filename or Path(filename).name != filename:
                raise typer.BadParameter(f"unsafe manifest filename: {filename!r}")
            path = source / filename
            if not path.is_file():
                raise typer.BadParameter(f"manifest file is missing: {path}")
            typer.echo(f"INGESTING {filename}", err=True)
            reports.extend(pipeline.ingest(path, force=force))
    typer.echo(json.dumps([report.model_dump(mode="json") for report in reports], indent=2))


@app.command("index")
def index_command(
    config: ConfigOption = Path("configs/default.yaml"),
    baseline: Annotated[bool, typer.Option("--baseline")] = False,
) -> None:
    """Build and persist the configured retrieval indexes."""
    target = _pipeline(config).build_indexes(baseline=baseline)
    typer.echo(str(target))


@app.command()
def retrieve(
    query: str,
    config: ConfigOption = Path("configs/default.yaml"),
    k: Annotated[int, typer.Option(min=1)] = 5,
    baseline: Annotated[bool, typer.Option("--baseline")] = False,
) -> None:
    """Print ranked, modality-tagged evidence as JSON."""
    result = _pipeline(config).retrieve(query, k=k, baseline=baseline)
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def ask(
    query: str,
    config: ConfigOption = Path("configs/default.yaml"),
    k: Annotated[int, typer.Option(min=1)] = 5,
    baseline: Annotated[bool, typer.Option("--baseline")] = False,
) -> None:
    """Retrieve evidence and run citation-gated generation."""
    answer = _pipeline(config).ask(query, k=k, baseline=baseline)
    typer.echo(answer.model_dump_json(indent=2))


@app.command()
def eval(
    gold: Annotated[Path, typer.Option("--gold", exists=True, dir_okay=False)] = Path("data/eval/fixture_gold.jsonl"),
    config: ConfigOption = Path("configs/default.yaml"),
    run_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Evaluate baseline and multimodal systems on a validated JSONL gold set."""
    pipeline = _pipeline(config)
    cases = [
        EvaluationCase.model_validate_json(line)
        for line in gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def system(baseline: bool) -> Callable[[EvaluationCase], SystemOutput]:
        def run(case: EvaluationCase) -> SystemOutput:
            answer = pipeline.ask(case.question, baseline=baseline, k=10)
            return SystemOutput(
                ranked_chunk_ids=[item.chunk.chunk_id for item in answer.evidence],
                answer=answer.text,
                grounded=answer.grounded,
                citation_valid=answer.grounded,
                latency_ms=answer.latency_ms,
                token_usage=answer.token_usage,
            )

        return run

    runner = EvaluationRunner(pipeline.config.paths.reports, seed=pipeline.config.seed)
    output = runner.run(
        cases,
        {"baseline": system(True), "multimodal": system(False)},
        config=pipeline.config.model_dump(mode="json"),
        run_id=run_id,
    )
    typer.echo(str(output))


@app.command()
def modalities() -> None:
    """List valid modality filter values."""
    typer.echo("\n".join(item.value for item in Modality))


if __name__ == "__main__":
    app()
