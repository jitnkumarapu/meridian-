"""Run the API and Vite UI together, forwarding termination to both children."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MMRAG API and React UI")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required; install Node.js 20+ before running the React UI")
    python = sys.executable
    api_environment = os.environ.copy()
    api_environment["MMRAG_CONFIG"] = str(args.config.resolve())
    processes = [
        subprocess.Popen(
            [python, "-m", "uvicorn", "mmrag.api.app:create_app", "--factory", "--port", "8010"],
            cwd=root,
            env=api_environment,
        ),
        subprocess.Popen(
            [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5177"],
            cwd=root / "frontend",
        ),
    ]

    def stop(_signum: int, _frame: object) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        failed = next(process.wait() for process in processes)
        if failed:
            raise SystemExit(failed)
    finally:
        stop(signal.SIGTERM, None)


if __name__ == "__main__":
    main()
