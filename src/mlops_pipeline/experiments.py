"""Log comparison runs to MLflow (not registered): baseline and recency experiments."""

import argparse

import mlflow
import pandas as pd

from mlops_pipeline.baseline import build_model as build_baseline
from mlops_pipeline.baseline import make_features
from mlops_pipeline.config import Config, load_config
from mlops_pipeline.gbm import (
    align_categoricals,
    build_lgbm,
    fit_lgbm,
    split_xy,
    trim_train,
)
from mlops_pipeline.metrics import evaluate
from mlops_pipeline.split import time_split
from mlops_pipeline.tracking import (
    DVC_POINTERS,
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


def fit_and_score(run_cfg: dict, cfg: Config, train: pd.DataFrame, val: pd.DataFrame, seed: int):
    """Fit one config on the (possibly trimmed) train window; return scores and details."""
    train = trim_train(
        train,
        run_cfg["gap_days"],
        run_cfg["trim_early_days"],
        cfg.split.train_end_day,
    )
    if run_cfg["model"] == "logreg":
        model = build_baseline().fit(make_features(train), train["isFraud"])
        return model.predict_proba(make_features(val))[:, 1], None, len(train)
    train_x, train_y = split_xy(train)
    val_x, val_y = split_xy(val)
    train_x, val_x = align_categoricals(train_x, val_x, min_count=cfg.model.min_category_count)
    model = fit_lgbm(
        build_lgbm(cfg.model, seed),
        train_x,
        train_y,
        val_x,
        val_y,
        cfg.model.early_stopping_rounds,
    )
    return predict_scores(model, val_x), model.best_iteration_, len(train)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="default: the config's seed")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg.model.seed if args.seed is None else args.seed

    df = pd.read_parquet(cfg.data.merged_path)
    # the test window is never used here
    train, val, _test = time_split(df, cfg.split.train_end_day, cfg.split.val_end_day)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    sha, dirty = git_state()
    lineage = {k: dvc_md5(p) for k, p in DVC_POINTERS.items()}

    for name in args.configs:
        run_cfg = CONFIGS[name]
        scores, best_iter, n_train = fit_and_score(run_cfg, cfg, train, val, seed)
        y = val["isFraud"]
        metrics: dict[str, float] = {}
        for budget in cfg.evaluation.budgets:
            metrics.update(evaluate(y, scores, budget))
        is_lgbm = run_cfg["model"] == "lgbm"
        run_name = f"{name}-seed{seed}" if is_lgbm else name
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
                    **run_cfg,
                    "features": "raw" if is_lgbm else "baseline",
                    "train_rows": n_train,
                    "val_rows": len(val),
                    "train_end_day": cfg.split.train_end_day,
                    "val_end_day": cfg.split.val_end_day,
                    **lineage,
                }
            )
            if is_lgbm:
                mlflow.log_param("seed", seed)
                mlflow.log_metric("best_iteration", best_iter)
            mlflow.log_metrics({f"val_{metric_key(k)}": v for k, v in metrics.items()})
        print(f"{run_name:18s} rows={n_train:7d} pr_auc={metrics['pr_auc']:.4f} dirty={dirty}")


if __name__ == "__main__":
    main()
