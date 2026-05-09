#!/usr/bin/env python3
"""Run all implemented experiment series 3 scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

SCRIPTS = [
    "run_experiment_01_signal_validation.py",
    "run_experiment_02_min_detectable_contrast.py",
    "run_experiment_03_noise_influence.py",
    "run_experiment_04_filtering.py",
    "run_experiment_05_spatial_resolution.py",
    "run_experiment_06_temporal_dynamics.py",
    "run_experiment_07_detector_comparison.py",
    "run_experiment_08_summary.py",
]


def main() -> None:
    base = Path(__file__).resolve().parent
    for script in SCRIPTS:
        cmd = [str(PYTHON), str(base / script)]
        print("Running", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
