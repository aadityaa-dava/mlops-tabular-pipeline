"""Inject controlled shifts into a clean window to see where the drift rules start to fire.

Runs in the monitoring environment, with the same reference sample, watchlist and thresholds
as run_monitor.py. The base window is the validation period: ordinary traffic that raises no
alert. Only features are used (never labels), and the test window is not touched.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from drift import evaluate_window, load_watchlist
from run_monitor import alert_reason, rounded

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
SECONDS_PER_DAY = 86_400
UNSEEN_CATEGORY = "new-domain.example"
HEADER = (
    f"{'scenario':44s} {'target':>7s} {'miss.shift':>10s} "
    f"{'detected':>8s} {'fires on':>8s} {'worst other':>11s}"
)


def scale_column(df: pd.DataFrame, col: str, factor: float) -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col] * factor
    return out


def shift_mean(df: pd.DataFrame, col: str, n_std: float, std: float) -> pd.DataFrame:
    """Add n_std standard deviations to every present value; missing stays missing."""
    out = df.copy()
    out[col] = out[col] + n_std * std
    return out


def blank_fraction(df: pd.DataFrame, col: str, frac: float, seed: int = 0) -> pd.DataFrame:
    """Make an exact share of the currently present values missing."""
    out = df.copy()
    present = out.index[out[col].notna()]
    k = int(round(frac * len(present)))
    chosen = np.random.default_rng(seed).choice(present, size=k, replace=False)
    out.loc[chosen, col] = np.nan
    return out


def replace_fraction(
    df: pd.DataFrame, col: str, frac: float, value: str, seed: int = 0
) -> pd.DataFrame:
    """Overwrite an exact share of rows with one value."""
    out = df.copy()
    k = int(round(frac * len(out)))
    chosen = np.random.default_rng(seed).choice(out.index, size=k, replace=False)
    out.loc[chosen, col] = value
    return out


def first_detected(rows: list[dict]) -> float | None:
    """Smallest magnitude at which the target column itself trips a rule, or None."""
    for row in sorted(rows, key=lambda r: r["magnitude"]):
        if row["target_detected"]:
            return row["magnitude"]
    return None


def build_scenarios(sim: dict, reference: pd.DataFrame, seed: int) -> list[dict]:
    scenarios = []
    col = sim["scale"]["column"]
    for factor in sim["scale"]["factors"]:
        scenarios.append(
            {
                "family": "scale",
                "column": col,
                "magnitude": factor,
                "label": f"{col} x{factor}",
                "apply": lambda d, c=col, f=factor: scale_column(d, c, f),
            }
        )
    col = sim["mean_shift"]["column"]
    std = float(reference[col].std())
    for k in sim["mean_shift"]["std_multiples"]:
        scenarios.append(
            {
                "family": "mean_shift",
                "column": col,
                "magnitude": k,
                "label": f"{col} + {k} std",
                "apply": lambda d, c=col, k=k, s=std: shift_mean(d, c, k, s),
            }
        )
    col = sim["blank"]["column"]
    for frac in sim["blank"]["fractions"]:
        scenarios.append(
            {
                "family": "blank",
                "column": col,
                "magnitude": frac,
                "label": f"{col}: {frac:.0%} of values go missing",
                "apply": lambda d, c=col, f=frac: blank_fraction(d, c, f, seed),
            }
        )
    col = sim["new_category"]["column"]
    for frac in sim["new_category"]["fractions"]:
        scenarios.append(
            {
                "family": "new_category",
                "column": col,
                "magnitude": frac,
                "label": f"{col}: {frac:.0%} rows get a new value",
                "apply": lambda d, c=col, f=frac: replace_fraction(d, c, f, UNSEEN_CATEGORY, seed),
            }
        )
    return scenarios


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    sim = cfg["simulation"]
    columns, categorical = load_watchlist(ROOT / cfg["export_dir"], cfg["top_n_features"])
    targets = {sim[k]["column"] for k in ("scale", "mean_shift", "blank", "new_category")}
    unknown = sorted(targets - set(columns))
    if unknown:
        raise SystemExit(f"simulation targets not in the watchlist: {unknown}")

    df = pd.read_parquet(ROOT / cfg["data_path"], columns=["TransactionDT", *columns])
    day = df["TransactionDT"] / SECONDS_PER_DAY
    ref_cfg = cfg["reference"]
    train = df[(day >= ref_cfg["start_day"]) & (day < ref_cfg["end_day"])]
    reference = train.sample(ref_cfg["sample_rows"], random_state=ref_cfg["seed"])[columns]
    win = sim["base_window"]
    window = df[(day >= win["start_day"]) & (day < win["end_day"])]
    n_rows = min(cfg["current"]["sample_rows"], len(window))
    base = window.sample(n_rows, random_state=cfg["current"]["seed"])[columns]
    th = cfg["thresholds"]

    def evaluate(current: pd.DataFrame) -> dict:
        return evaluate_window(
            reference, current, columns, categorical, th["column_score"], th["missing_rate_shift"]
        )

    baseline = evaluate(base)
    print(f"pandas {pd.__version__} | base window: {win['name']} ({len(base)} rows)")
    print("baseline scores:", {c: round(s, 3) for c, s in baseline["scores"].items()})
    print(f"baseline fires on: {alert_reason(baseline)}\n")

    rows = []
    print(HEADER)
    for sc in build_scenarios(sim, reference, sim["seed"]):
        res = evaluate(sc["apply"](base))
        col = sc["column"]
        others = [v for c, v in res["scores"].items() if c != col]
        row = {
            "family": sc["family"],
            "column": col,
            "magnitude": sc["magnitude"],
            "label": sc["label"],
            "target_score": res["scores"][col],
            "target_missing_shift": res["missing_shifts"][col],
            "target_detected": col in res["drifted_columns"] or col in res["missing_shift_columns"],
            "fires_on": alert_reason(res),
            "worst_other_score": max(others),
        }
        rows.append(row)
        print(
            f"{row['label']:44s} {row['target_score']:7.3f} {row['target_missing_shift']:10.3f} "
            f"{row['target_detected']!s:>8s} {row['fires_on']:>8s} {row['worst_other_score']:11.3f}"
        )

    print("\nsmallest shift where the target column trips a rule:")
    summary = {}
    for family in dict.fromkeys(r["family"] for r in rows):
        family_rows = [r for r in rows if r["family"] == family]
        found = first_detected(family_rows)
        summary[family] = found
        text = "not detected in the range tested" if found is None else found
        print(f"  {family} ({family_rows[0]['column']}): {text}")

    report = {"thresholds": th, "base_window": win, "summary": summary, "rows": rows}
    out = ROOT / sim["report_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rounded(report), indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
