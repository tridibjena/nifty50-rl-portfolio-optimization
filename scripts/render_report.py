#!/usr/bin/env python3
"""Re-render figures and RESULTS.md from saved run artefacts, without retraining.

The full pipeline retrains PPO across every walk-forward window, which takes about twenty
minutes. Iterating on presentation should not cost that. This reads what
``run_pipeline.py`` already wrote to ``results/`` and regenerates the derived views.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

warnings.filterwarnings("ignore")

from nifty_rl.report import figures
from nifty_rl.report.build import write_results

ASSETS = PROJECT_ROOT / "assets" / "v2"
RESULTS = PROJECT_ROOT / "results"


def main() -> None:
    per_window = pd.read_csv(RESULTS / "walk_forward_windows.csv")
    summary = pd.read_csv(RESULTS / "run_summary.csv", index_col=0).iloc[:, 0].to_dict()

    for mode in ("light", "dark"):
        suffix = "" if mode == "light" else "_dark"
        figures.save(
            *figures.ppo_seed_dispersion(per_window, mode=mode),
            ASSETS / f"ppo_seed_dispersion{suffix}.png",
        )
    print(f"figures -> {ASSETS}")

    path = write_results(RESULTS, ASSETS, summary, PROJECT_ROOT / "RESULTS.md")
    print(f"report  -> {path}")


if __name__ == "__main__":
    main()
