"""Compare time windows of the merged data with a training-period reference."""

import json
from pathlib import Path

import pandas as pd
import yaml

from drift import evaluate_window, load_watchlist

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
SECONDS_PER_DAY = 86_400


def rounded(obj, ndigits: int = 6):
    """Round every float inside nested dicts and lists, for a readable report."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: rounded(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v, ndigits) for v in obj]
    return obj


def alert_reason(res: dict) -> str:
    """Which rule fired for a window: 'score', 'missing', 'both' or '-'."""
    score, missing = bool(res["drifted_columns"]), bool(res["missing_shift_columns"])
    if score and missing:
        return "both"
    return "score" if score else "missing" if missing else "-"


def print_tables(results: dict, columns: list[str]) -> None:
    names = " ".join(f"{c[:8]:>8s}" for c in columns)
    print("drift scores (Evidently distance per column):")
    print(f"{'window':36s} fires on {names}")
    for label, res in results.items():
        cells = " ".join(f"{res['scores'][c]:8.3f}" for c in columns)
        print(f"{label:36s} {alert_reason(res):8s} {cells}")
    print("\nchange in share of missing values:")
    print(f"{'window':36s}       {names}")
    for label, res in results.items():
        cells = " ".join(f"{res['missing_shifts'][c]:8.3f}" for c in columns)
        print(f"{label:36s}       {cells}")


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    columns, categorical = load_watchlist(ROOT / cfg["export_dir"], cfg["top_n_features"])
    df = pd.read_parquet(ROOT / cfg["data_path"], columns=["TransactionDT", *columns])
    day = df["TransactionDT"] / SECONDS_PER_DAY
    print(f"pandas {pd.__version__} | rows {len(df)} | watching {len(columns)} columns")

    ref = cfg["reference"]
    train = df[(day >= ref["start_day"]) & (day < ref["end_day"])]
    reference = train.sample(ref["sample_rows"], random_state=ref["seed"])
    thresholds = cfg["thresholds"]

    def check(ref_frame: pd.DataFrame, cur_frame: pd.DataFrame) -> dict:
        return evaluate_window(
            ref_frame[columns],
            cur_frame[columns],
            columns,
            categorical,
            thresholds["column_score"],
            thresholds["missing_rate_shift"],
        )

    null = cfg["null_check"]
    half_a = train.sample(null["sample_rows"], random_state=null["seeds"][0])
    half_b = train.drop(half_a.index).sample(null["sample_rows"], random_state=null["seeds"][1])
    results = {"null: two random halves of train": check(half_a, half_b)}

    current_cfg = cfg["current"]
    for window in cfg["windows"]:
        cur = df[(day >= window["start_day"]) & (day < window["end_day"])]
        n_rows = min(current_cfg["sample_rows"], len(cur))
        cur = cur.sample(n_rows, random_state=current_cfg["seed"])
        results[window["name"]] = check(reference, cur)

    print_tables(results, columns)
    report = {
        "watched_columns": columns,
        "categorical_columns": categorical,
        "thresholds": thresholds,
        "reference": {**ref, "rows": len(reference)},
        "windows": results,
    }
    out = ROOT / cfg["report_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rounded(report), indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
