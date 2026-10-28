"""Compare time windows of the merged data with a training-period reference."""

import json
from pathlib import Path

import pandas as pd
import yaml

from drift import (
    calibrate_thresholds,
    judge,
    load_watchlist,
    measure_window,
    presence_rates,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
SECONDS_PER_DAY = 86_400
NULL_LABEL = "null: two random halves of train"


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


def fired(res: dict) -> str:
    """Columns that tripped a rule, e.g. 'D2,V258(miss)'."""
    names = list(res["drifted_columns"]) + [f"{c}(miss)" for c in res["missing_shift_columns"]]
    return ",".join(names) or "-"


def verdict(res: dict) -> dict:
    return {k: res[k] for k in ("drifted_columns", "missing_shift_columns", "alert")}


def sample_window(df: pd.DataFrame, day: pd.Series, window: dict, sample_rows: int, seed: int):
    """Rows of one time window, sampled down to sample_rows."""
    rows = df[(day >= window["start_day"]) & (day < window["end_day"])]
    return rows.sample(min(sample_rows, len(rows)), random_state=seed)


def print_presence(columns, reference_rates, block_rates, later) -> None:
    print("share of rows where each column is present:")
    headers = ["reference", "blocks min", "blocks max", *later]
    print(f"{'column':16s} " + " ".join(f"{h:>10s}" for h in headers))
    for c in columns:
        lo = min(r[c] for r in block_rates)
        hi = max(r[c] for r in block_rates)
        values = [reference_rates[c], lo, hi, *[rates[c] for rates in later.values()]]
        print(f"{c:16s} " + " ".join(f"{v:10.3f}" for v in values))


def print_limits(columns, limits, normal) -> None:
    print("per-column limits, calibrated on the training blocks:")
    headers = ["score limit", "largest seen", "missing limit", "largest seen"]
    print(f"{'column':16s} " + " ".join(f"{h:>13s}" for h in headers))
    for c in columns:
        top_score = max(w["scores"][c] for w in normal)
        top_shift = max(w["missing_shifts"][c] for w in normal)
        values = [limits[c]["score"], top_score, limits[c]["missing_shift"], top_shift]
        print(f"{c:16s} " + " ".join(f"{v:13.3f}" for v in values))


def print_windows(results: dict) -> None:
    print(f"{'window':36s} {'global':>8s} {'per-col':>8s}  fires (per-column rule)")
    for label, both in results.items():
        g, p = both["global"], both["per_column"]
        print(f"{label:36s} {alert_reason(g):>8s} {alert_reason(p):>8s}  {fired(p)}")


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    columns, categorical = load_watchlist(ROOT / cfg["export_dir"], cfg["top_n_features"])
    df = pd.read_parquet(ROOT / cfg["data_path"], columns=["TransactionDT", *columns])
    day = df["TransactionDT"] / SECONDS_PER_DAY
    print(f"pandas {pd.__version__} | rows {len(df)} | watching {len(columns)} columns\n")

    ref = cfg["reference"]
    train = df[(day >= ref["start_day"]) & (day < ref["end_day"])]
    reference = train.sample(ref["sample_rows"], random_state=ref["seed"])

    null = cfg["null_check"]
    half_a = train.sample(null["sample_rows"], random_state=null["seeds"][0])
    half_b = train.drop(half_a.index).sample(null["sample_rows"], random_state=null["seeds"][1])
    measured = {NULL_LABEL: measure_window(half_a[columns], half_b[columns], columns, categorical)}

    frames = {}
    current_cfg = cfg["current"]
    for window in cfg["windows"]:
        name = window["name"]
        n_rows, seed = current_cfg["sample_rows"], current_cfg["seed"]
        frames[name] = sample_window(df, day, window, n_rows, seed)
        measured[name] = measure_window(
            reference[columns], frames[name][columns], columns, categorical
        )

    calibrate = [w["name"] for w in cfg["windows"] if w.get("calibrate")]
    normal = [measured[name] for name in calibrate]
    cal = cfg["calibration"]
    limits = calibrate_thresholds(
        normal, columns, cal["safety_factor"], cal["score_floor"], cal["missing_shift_floor"]
    )
    score_limits = {c: t["score"] for c, t in limits.items()}
    missing_limits = {c: t["missing_shift"] for c, t in limits.items()}
    glob = cfg["thresholds"]
    results = {
        label: {
            "global": judge(m, glob["column_score"], glob["missing_rate_shift"]),
            "per_column": judge(m, score_limits, missing_limits),
        }
        for label, m in measured.items()
    }

    block_rates = [presence_rates(frames[n], columns) for n in calibrate]
    later = {
        w["name"].split()[0]: presence_rates(frames[w["name"]], columns)
        for w in cfg["windows"]
        if not w.get("calibrate")
    }
    print_presence(columns, presence_rates(reference, columns), block_rates, later)
    print()
    print_limits(columns, limits, normal)
    print()
    print_windows(results)

    report = {
        "watched_columns": columns,
        "categorical_columns": categorical,
        "reference": {**ref, "rows": len(reference)},
        "global_thresholds": glob,
        "calibration": {**cal, "windows": calibrate, "limits": limits},
        "windows": {
            label: {
                "rows": m["rows"],
                "scores": m["scores"],
                "missing_shifts": m["missing_shifts"],
                "global": verdict(results[label]["global"]),
                "per_column": verdict(results[label]["per_column"]),
            }
            for label, m in measured.items()
        },
    }
    out = ROOT / cfg["report_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rounded(report), indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
