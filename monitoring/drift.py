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


def presence_rates(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Share of rows in which each column has a value."""
    return {c: float(frame[c].notna().mean()) for c in columns}


def measure_window(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    categorical: list[str],
) -> dict:
    """The raw measurements for one window: drift scores and missing-share shifts."""
    return {
        "rows": len(current),
        "scores": drift_scores(reference, current, columns, categorical),
        "missing_shifts": missing_rate_shifts(reference, current, columns),
    }


def limit_for(threshold: float | dict[str, float], column: str) -> float:
    """A threshold is either one number for every column or a limit per column."""
    return threshold[column] if isinstance(threshold, dict) else threshold


def judge(
    measured: dict,
    score_threshold: float | dict[str, float],
    missing_threshold: float | dict[str, float],
) -> dict:
    """Apply thresholds to a measurement and flag the columns that reach them."""
    drifted = sorted(c for c, s in measured["scores"].items() if s >= limit_for(score_threshold, c))
    shifted = sorted(
        c for c, s in measured["missing_shifts"].items() if s >= limit_for(missing_threshold, c)
    )
    return {
        **measured,
        "drifted_columns": drifted,
        "missing_shift_columns": shifted,
        "alert": bool(drifted or shifted),
    }


def evaluate_window(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    categorical: list[str],
    score_threshold: float | dict[str, float],
    missing_threshold: float | dict[str, float],
) -> dict:
    """Measure one window and judge it against the given thresholds."""
    measured = measure_window(reference, current, columns, categorical)
    return judge(measured, score_threshold, missing_threshold)


def calibrate_thresholds(
    normal_windows: list[dict],
    columns: list[str],
    safety_factor: float,
    score_floor: float,
    missing_floor: float,
) -> dict[str, dict[str, float]]:
    """Per-column limits: max(floor, safety_factor * the largest value seen in normal windows)."""
    if not normal_windows:
        raise ValueError("need at least one normal window to calibrate on")
    limits = {}
    for c in columns:
        top_score = max(w["scores"][c] for w in normal_windows)
        top_shift = max(w["missing_shifts"][c] for w in normal_windows)
        limits[c] = {
            "score": max(score_floor, safety_factor * top_score),
            "missing_shift": max(missing_floor, safety_factor * top_shift),
        }
    return limits
