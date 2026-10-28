import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import drift
from drift import drift_scores, evaluate_window, load_watchlist, missing_rate_shifts
from run_monitor import rounded

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ["amount", "n_items", "card", "sparse"]
CATEGORICAL = ["card"]
THRESHOLDS = {"score_threshold": 0.2, "missing_threshold": 0.15}


def make_frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount": rng.lognormal(4, 1, n),
            "n_items": rng.poisson(3, n).astype(float),
            "card": rng.choice(["visa", "mastercard", None], n, p=[0.6, 0.3, 0.1]),
            "sparse": np.where(rng.random(n) < 0.4, np.nan, rng.normal(size=n)),
        }
    )


def evaluate(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    return evaluate_window(reference, current, COLUMNS, CATEGORICAL, **THRESHOLDS)


def test_watchlist_takes_top_features_and_flags_categoricals(tmp_path):
    (tmp_path / "feature_importance.json").write_text(
        json.dumps({"low": 0.1, "top": 0.6, "mid": 0.3})
    )
    (tmp_path / "categories.json").write_text(json.dumps({"mid": ["a"], "unused": ["b"]}))
    columns, categorical = load_watchlist(tmp_path, top_n=2)
    assert columns == ["top", "mid"]
    assert categorical == ["mid"]


def test_same_distribution_does_not_alert():
    result = evaluate(make_frame(seed=0), make_frame(seed=1))
    assert result["drifted_columns"] == []
    assert result["missing_shift_columns"] == []
    assert not result["alert"]


def test_scaled_numeric_column_alerts_and_is_named():
    current = make_frame(seed=1)
    current["amount"] = current["amount"] * 1.8
    result = evaluate(make_frame(seed=0), current)
    assert result["alert"]
    assert result["drifted_columns"] == ["amount"]


def test_category_mix_change_alerts():
    current = make_frame(seed=1)
    rng = np.random.default_rng(5)
    current["card"] = rng.choice(["visa", "mastercard", None], len(current), p=[0.1, 0.8, 0.1])
    result = evaluate(make_frame(seed=0), current)
    assert "card" in result["drifted_columns"]


def test_a_field_that_stops_arriving_is_flagged_by_the_missingness_check():
    current = make_frame(seed=1)
    rng = np.random.default_rng(6)
    current["sparse"] = np.where(rng.random(len(current)) < 0.9, np.nan, current["sparse"])
    result = evaluate(make_frame(seed=0), current)
    assert "sparse" in result["missing_shift_columns"]
    assert result["alert"]


def test_missing_rate_shifts_are_exact():
    reference = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]})
    current = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})
    assert missing_rate_shifts(reference, current, ["a"]) == {"a": 0.25}


def test_unexpected_evidently_output_is_rejected(monkeypatch):
    class EmptyReport:
        def run(self, current, reference):
            return self

        def dict(self):
            return {"metrics": [], "tests": []}

    monkeypatch.setattr(drift, "Report", lambda metrics: EmptyReport())
    with pytest.raises(RuntimeError, match="no drift score"):
        drift_scores(make_frame(50), make_frame(50, seed=1), COLUMNS, CATEGORICAL)


def test_windows_line_up_with_the_training_split():
    split = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())["split"]
    monitor = yaml.safe_load((ROOT / "monitoring" / "config.yaml").read_text())
    windows = {w["name"].split()[0]: w for w in monitor["windows"]}
    assert monitor["reference"]["end_day"] == split["train_end_day"]
    assert windows["validation"]["start_day"] == split["train_end_day"]
    assert windows["validation"]["end_day"] == split["val_end_day"]
    assert windows["test"]["start_day"] == split["val_end_day"]


def test_rounded_handles_nested_structures():
    assert rounded({"a": [0.123456789, {"b": 1}]}, 3) == {"a": [0.123, {"b": 1}]}


def test_alert_reason_names_the_rule_that_fired():
    from run_monitor import alert_reason

    base = {"drifted_columns": [], "missing_shift_columns": []}
    assert alert_reason(base) == "-"
    assert alert_reason({**base, "drifted_columns": ["a"]}) == "score"
    assert alert_reason({**base, "missing_shift_columns": ["a"]}) == "missing"
    assert alert_reason({"drifted_columns": ["a"], "missing_shift_columns": ["b"]}) == "both"
