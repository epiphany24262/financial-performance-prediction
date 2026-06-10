from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PROJECT_ROOT as ROOT
from src.io_utils import write_csv
from src.plots import (
    plot_blend_weights,
    plot_model_comparison,
    plot_oof_scatter,
    plot_residual_distribution,
    plot_target_score_heatmap,
)


SCORE_TABLES = [
    ROOT / "results" / "tables" / "baseline_scores.csv",
    ROOT / "results" / "tables" / "sklearn_model_scores.csv",
    ROOT / "results" / "tables" / "catboost_model_scores.csv",
    ROOT / "results" / "tables" / "final_model_scores.csv",
]


def main() -> None:
    frames = [pd.read_csv(path) for path in SCORE_TABLES if path.exists()]
    if not frames:
        raise FileNotFoundError("No score tables found for final figures")
    scores = pd.concat(frames, ignore_index=True)
    scores = scores.drop_duplicates(subset=["experiment_id"], keep="last")
    if "M3d_catboost_direct_history_metadata_engineered" in set(scores["experiment_id"]):
        scores = scores[
            scores["experiment_id"] != "M3_catboost_direct_history_metadata_engineered"
        ].copy()
    write_csv(ROOT / "results" / "tables" / "all_model_scores.csv", scores)

    figures = ROOT / "figures"
    plot_model_comparison(scores, figures / "fig09_model_comparison.png")
    plot_target_score_heatmap(scores, figures / "fig10_target_score_heatmap.png")
    plot_oof_scatter(pd.read_csv(ROOT / "results" / "oof" / "m6_oof_blend.csv"), figures / "fig11_oof_scatter.png")
    plot_residual_distribution(
        pd.read_csv(ROOT / "results" / "oof" / "m6_oof_blend.csv"),
        figures / "fig12_residual_distribution.png",
    )
    plot_blend_weights(pd.read_csv(ROOT / "results" / "tables" / "blend_scores.csv"), figures / "fig13_blend_weights.png")
    print("Saved final model figures to figures/.")


if __name__ == "__main__":
    main()
