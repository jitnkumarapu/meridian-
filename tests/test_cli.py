from typer.testing import CliRunner

from mmrag.cli import app


def test_modalities_command() -> None:
    result = CliRunner().invoke(app, ["modalities"])
    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["text", "table", "figure"]
