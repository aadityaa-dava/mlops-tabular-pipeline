"""Drift checks between a reference window and a current window.

Runs in the monitoring environment (pandas 2.x + Evidently), which does not import the
main package. It reads the model export folder as plain files: feature_importance.json
(which columns matter) and categories.json (which of them are categorical).
"""

import json
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset


def load_watchlist(export_dir: Path | str, top_n: int) -> tuple[list[str], list[str]]:
    """The top_n features by gain share, and which of them the model treats as categorical."""
    folder = Path(export_dir)
    shares = json.loads((folder / "feature_importance.json").read_text())
    categories = json.loads((folder / "categories.json").read_text())
    columns = sorted(shares, key=shares.get, reverse=True)[:top_n]
    return columns, [c for c in columns if c in categories]


def drift_scores(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    categorical: list[str],
) -> dict[str, float]:
    """Evidently's per-column drift score: a distance, higher means more drift."""
    numerical = [c for c in columns if c not in categorical]
    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    ref_ds = Dataset.from_pandas(
        reference[columns].reset_index(drop=True), data_definition=definition
    )
    cur_ds = Dataset.from_pandas(
        current[columns].reset_index(drop=True), data_definition=definition
    )
    result = Report([DataDriftPreset()]).run(cur_ds, ref_ds).dict()
    scores = {}
    for metric in result["metrics"]:
        config = metric["config"]
        if config["type"].endswith(":ValueDrift"):
            scores[config["column"]] = float(metric["value"])
    missing = sorted(set(columns) - set(scores))
    if missing:
        raise RuntimeError(f"no drift score returned for {missing}")
    return scores


def missing_rate_shifts(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> dict[str, float]:
    """Absolute change in each column's share of missing values."""
    return {
        c: abs(float(current[c].isna().mean()) - float(reference[c].isna().mean())) for c in columns
    }


def evaluate_window(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    categorical: list[str],
    score_threshold: float,
    missing_threshold: float,
) -> dict:
    """Scores, missingness shifts and an alert flag for one current window."""
    scores = drift_scores(reference, current, columns, categorical)
    shifts = missing_rate_shifts(reference, current, columns)
    drifted = sorted(c for c, s in scores.items() if s >= score_threshold)
    shifted = sorted(c for c, s in shifts.items() if s >= missing_threshold)
    return {
        "rows": len(current),
        "scores": scores,
        "missing_shifts": shifts,
        "drifted_columns": drifted,
        "missing_shift_columns": shifted,
        "alert": bool(drifted or shifted),
    }
