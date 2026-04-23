"""Tests for the command-line QASM runner."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
RUNNER = REPO_ROOT / "run_qasm.py"


def test_run_qasm_cli_noiseless_outputs_state_and_counts():
    """The CLI should print the statevector and counts for a simple circuit."""
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(EXAMPLES_DIR / "bell_state.qasm"),
            "--shots",
            "128",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Pre-measurement statevector:" in completed.stdout
    assert "Counts:" in completed.stdout
    assert "bell_state.qasm" in completed.stdout


def test_run_qasm_cli_can_save_plots(tmp_path):
    """The CLI should save plot files when asked."""
    output_dir = tmp_path / "plots"
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(EXAMPLES_DIR / "parametric_gates.qasm"),
            "--shots",
            "64",
            "--save-plots",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "counts.png").exists()
    assert (output_dir / "probabilities.png").exists()
    assert (output_dir / "bloch_q0.png").exists()
