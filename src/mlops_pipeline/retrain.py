"""Train a candidate on newer data and compare it with the champion on the same rows.

The candidate is written to artifacts/candidate/ only if the paired promotion rule accepts it.
Nothing here registers a model, overwrites the champion, or touches the deployed image: replacing
the champion is a separate, deliberate step.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from mlops_pipeline.config import load_config
from mlops_pipeline.gbm import align_categoricals, build_lgbm, fit_lgbm, split_xy
from mlops_pipeline.inference import apply_categories, predict_scores
from mlops_pipeline.metrics import evaluate
from mlops_pipeline.model_store import DEFAULT_MODEL_DIR, load_exported, write_export
from mlops_pipeline.promotion import PAIRED_MARGIN, should_promote_paired
from mlops_pipeline.tracking import fit_categories

SECONDS_PER_DAY = 86_400
CANDIDATE_DIR = "artifacts/candidate"
REPORT_PATH = "reports/retrain_report.json"
N_BOOTSTRAP = 200


def paired_bootstrap(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
) -> tuple[float, float, float]:
    """PR-AUC of a minus b on the same rows: (difference, interval low, interval high)."""
    diff = average_precision_score(y, scores_a) - average_precision_score(y, scores_b)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boots.append(
            average_precision_score(y[idx], scores_a[idx])
            - average_precision_score(y[idx], scores_b[idx])
        )
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(diff), float(low), float(high)


def recall_drop(y: np.ndarray, candidate: np.ndarray, champion: np.ndarray, budget: float) -> float:
    """Champion recall at the alert budget minus candidate recall (positive: candidate worse)."""
    key = f"recall_at_{budget:.0%}"
    return float(evaluate(y, champion, budget)[key] - evaluate(y, candidate, budget)[key])


def decision_report(
    diff: float,
    low: float,
    high: float,
    drop: float,
    roundtrip: float,
    meta: dict,
) -> dict:
    decision = should_promote_paired(diff, low, drop, roundtrip)
    return {
        "promote": decision.promote,
        "reason": decision.reason,
        "margin": PAIRED_MARGIN,
        "pr_auc_diff": diff,
        "interval": [low, high],
        "recall_at_5pct_drop": drop,
        "roundtrip_max_abs_diff": roundtrip,
        **meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--train-end-day", type=float, default=135.0)
    parser.add_argument("--compare-start-day", type=float, default=135.0)
    parser.add_argument("--compare-end-day", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=None, help="default: the config's seed")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.compare_end_day > cfg.split.val_end_day:
        raise SystemExit("the comparison window may not reach into the test window")
    if args.train_end_day > args.compare_start_day:
        raise SystemExit("the candidate may not train on rows it is compared on")

    df = pd.read_parquet(cfg.data.merged_path)
    day = df["TransactionDT"] / SECONDS_PER_DAY
    df = df[day < cfg.split.val_end_day]  # the test window is never loaded into the comparison
    day = df["TransactionDT"] / SECONDS_PER_DAY
    train = df[day < args.train_end_day]
    window = df[(day >= args.compare_start_day) & (day < args.compare_end_day)]

    champion = load_exported(DEFAULT_MODEL_DIR)
    x_win, y_win = split_xy(window)
    y = y_win.to_numpy()
    champ_scores = predict_scores(champion.model, apply_categories(x_win, champion.categories))

    x_train, y_train = split_xy(train)
    x_train, x_cmp = align_categoricals(x_train, x_win, min_count=cfg.model.min_category_count)
    seed = cfg.model.seed if args.seed is None else args.seed
    model = fit_lgbm(
        build_lgbm(cfg.model, seed),
        x_train,
        y_train,
        x_cmp,
        y_win,
        cfg.model.early_stopping_rounds,
    )
    cand_scores = predict_scores(model, x_cmp)

    diff, low, high = paired_bootstrap(y, cand_scores, champ_scores)
    drop = recall_drop(y, cand_scores, champ_scores, 0.05)
    categories = fit_categories(x_train)
    candidate_dir = Path(CANDIDATE_DIR)
    scratch = candidate_dir.parent / "candidate_check"
    write_export(
        scratch, model.booster_, categories, {"model_version": "candidate"}, model.best_iteration_
    )
    reloaded = load_exported(scratch)
    again = predict_scores(reloaded.model, apply_categories(x_win, reloaded.categories))
    roundtrip = float(np.abs(again - cand_scores).max())

    meta = {
        "candidate_train_days": [0, args.train_end_day],
        "compare_days": [args.compare_start_day, args.compare_end_day],
        "compare_rows": len(y),
        "compare_frauds": int(y.sum()),
        "candidate_train_rows": len(train),
        "candidate_trees": int(model.best_iteration_),
        "champion_version": champion.version,
        "champion_pr_auc": float(average_precision_score(y, champ_scores)),
        "candidate_pr_auc": float(average_precision_score(y, cand_scores)),
        "caveat": "early stopping used the comparison window, so the candidate is optimistic",
    }
    report = decision_report(diff, low, high, drop, roundtrip, meta)

    if report["promote"]:
        write_export(
            candidate_dir,
            model.booster_,
            categories,
            {"model_version": "candidate", **meta},
            model.best_iteration_,
        )
    out = Path(REPORT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"champion v{champion.version} PR-AUC {meta['champion_pr_auc']:.4f} | "
        f"candidate {meta['candidate_pr_auc']:.4f} on days "
        f"{args.compare_start_day:g}-{args.compare_end_day:g} ({len(y)} rows)"
    )
    print(
        f"paired diff {diff:+.4f}  interval {low:+.4f} to {high:+.4f}  "
        f"recall@5% drop {drop:+.4f}  round trip {roundtrip:.1e}"
    )
    print(f"decision: promote={report['promote']} ({report['reason']})")
    print(
        f"candidate written to {CANDIDATE_DIR}"
        if report["promote"]
        else "no candidate written; the champion is unchanged"
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
