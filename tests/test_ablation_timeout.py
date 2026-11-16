from __future__ import annotations

import pathlib
import subprocess
import sys


TIMEOUT = pathlib.Path(__file__).parents[1] / "ablations" / "bin" / "timeout"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(TIMEOUT), *args], text=True, capture_output=True, timeout=3)


def test_timeout_passes_exit_status() -> None:
    assert run("1", sys.executable, "-c", "raise SystemExit(7)").returncode == 7


def test_timeout_terminates_process_group() -> None:
    assert run("--kill-after=.1s", ".05s", sys.executable, "-c", "import time; time.sleep(2)").returncode == 124


def test_timeout_rejects_bad_input() -> None:
    assert run("nope", "true").returncode == 125
    assert run("--preserve-status", "1", "true").returncode == 125

