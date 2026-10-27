"""Log comparison runs to MLflow (not registered): baseline and recency experiments."""

import argparse

import lightgbm as lgb
import mlflow
import pandas as pd

from mlops_pipeline.baseline import build_model as build_baseline
from mlops_pipeline.baseline import make_features
from mlops_pipeline.gbm import (
    TRAIN_END_DAY,
    align_categoricals,
    build_lgbm,
    split_xy,
    trim_train,
)
from mlops_pipeline.metrics import evaluate
from mlops_pipeline.split import time_split
from mlops_pipeline.tracking import (
    DVC_POINTERS,
    EXPERIMENT,
    TRACKING_URI,
    VAL_END_DAY,
    dvc_md5,
    git_state,
    metric_key,
    predict_scores,
)

CONFIGS = {
    "baseline": {"model": "logreg", "gap_days": 0.0, "trim_early_days": 0.0},
    "gap14": {"model": "lgbm", "gap_days": 14.0, "trim_early_days": 0.0},
    "control14": {"model": "lgbm", "gap_days": 0.0, "trim_early_days": 14.0},
}


def fit_and_score(cfg: dict, train: pd.DataFrame, val: pd.DataFrame, seed: int):
    """Fit one config on the (possibly trimmed) train window; return scores and details."""
    train = trim_train(train, cfg["gap_days"], cfg["trim_early_days"])
    if cfg["model"] == "logreg":
        model = build_baseline().fit(make_features(train), train["isFraud"])
        return model.predict_proba(make_features(val))[:, 1], None, len(train)
    train_x, train_y = split_xy(train)
    val_x, val_y = split_xy(val)
    train_x, val_x = align_categoricals(train_x, val_x)
    model = build_lgbm(seed).fit(
        train_x,
        train_y,
        eval_X=val_x,
        eval_y=val_y,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return predict_scores(model, val_x), model.best_iteration_, len(train)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()

    df = pd.read_parquet("data/processed/train_merged.parquet")
    train, val, _test = time_split(df, TRAIN_END_DAY, VAL_END_DAY)  # test stays untouched
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    sha, dirty = git_state()
    lineage = {k: dvc_md5(p) for k, p in DVC_POINTERS.items()}

    for name in args.configs:
        cfg = CONFIGS[name]
        scores, best_iter, n_train = fit_and_score(cfg, train, val, args.seed)
        y = val["isFraud"]
        metrics = {**evaluate(y, scores, 0.01), **evaluate(y, scores, 0.05)}
        is_lgbm = cfg["model"] == "lgbm"
        run_name = f"{name}-seed{args.seed}" if is_lgbm else name
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "git_commit": sha,
                    "git_dirty": str(dirty),
                    "scored_on": "validation",
                    "registered": "no",
                }
            )
            mlflow.log_params(
                {
                    **cfg,
                    "features": "raw" if is_lgbm else "baseline",
                    "train_rows": n_train,
                    "val_rows": len(val),
                    "train_end_day": TRAIN_END_DAY,
                    "val_end_day": VAL_END_DAY,
                    **lineage,
                }
            )
            if is_lgbm:
                mlflow.log_param("seed", args.seed)
                mlflow.log_metric("best_iteration", best_iter)
            mlflow.log_metrics({f"val_{metric_key(k)}": v for k, v in metrics.items()})
        print(
            f"{run_name:18s} rows={n_train:7d} pr_auc={metrics['pr_auc']:.4f} "
            f"recall@5%={metrics['recall_at_5%']:.4f} dirty={dirty}"
        )


if __name__ == "__main__":
    main()
