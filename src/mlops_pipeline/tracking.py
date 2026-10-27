"""Train the raw-feature LightGBM from the config and log it to MLflow."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.lightgbm
import numpy as np
import pandas as pd

from mlops_pipeline.config import load_config
from mlops_pipeline.gbm import align_categoricals, build_lgbm, fit_lgbm, split_xy
from mlops_pipeline.metrics import evaluate
from mlops_pipeline.split import time_split

DVC_POINTERS = {
    "data_train_transaction_md5": "data/raw/train_transaction.csv.dvc",
    "data_train_identity_md5": "data/raw/train_identity.csv.dvc",
}


TRAIN_METRICS_PATH = Path("reports/train_metrics.json")


def metric_key(name: str) -> str:
    """MLflow metric names cannot contain '%'."""
    return name.replace("%", "pct")


def dvc_md5(path: str) -> str:
    """Read the dataset hash from a DVC pointer file."""
    match = re.search(r"md5:\s*([0-9a-f]{32})", Path(path).read_text())
    if match is None:
        raise ValueError(f"no md5 found in {path}")
    return match.group(1)


def git_state() -> tuple[str, bool]:
    """Current commit SHA and whether the working tree has uncommitted changes."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return sha, bool(status)


def fit_categories(x: pd.DataFrame) -> dict[str, list[str]]:
    """Category lists of the categorical columns of an already-aligned training frame."""
    return {
        c: [str(v) for v in x[c].cat.categories]
        for c in x.columns
        if isinstance(x[c].dtype, pd.CategoricalDtype)
    }


def apply_categories(x: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Cast text columns to the training categories; unseen or rare values become NaN."""
    out = x.copy()
    for col, cats in categories.items():
        out[col] = out[col].where(out[col].isin(cats)).astype(pd.CategoricalDtype(cats))
    return out


def predict_scores(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="default: the config's seed")
    parser.add_argument(
        "--no-register", action="store_true", help="log the run but do not register the model"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg.model.seed if args.seed is None else args.seed

    df = pd.read_parquet(cfg.data.merged_path)
    # the test window stays untouched
    train, val, _test = time_split(df, cfg.split.train_end_day, cfg.split.val_end_day)
    train_x, train_y = split_xy(train)
    val_x_raw, val_y = split_xy(val)
    train_x, val_x = align_categoricals(train_x, val_x_raw, min_count=cfg.model.min_category_count)
    categories = fit_categories(train_x)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    sha, dirty = git_state()
    model = build_lgbm(cfg.model, seed)
    register_as = None if args.no_register else cfg.mlflow.registered_model

    with mlflow.start_run(run_name=f"lgbm-raw-seed{seed}") as run:
        mlflow.set_tags(
            {
                "git_commit": sha,
                "git_dirty": str(dirty),
                "scored_on": "validation",
                "registered": "no" if args.no_register else "yes",
            }
        )
        mlflow.log_params(model.get_params())
        mlflow.log_params(
            {
                "seed": seed,
                "features": "raw",
                "train_end_day": cfg.split.train_end_day,
                "val_end_day": cfg.split.val_end_day,
                "train_rows": len(train),
                "val_rows": len(val),
                **{k: dvc_md5(p) for k, p in DVC_POINTERS.items()},
            }
        )

        fit_lgbm(model, train_x, train_y, val_x, val_y, cfg.model.early_stopping_rounds)
        scores = predict_scores(model, val_x)
        metrics: dict[str, float] = {}
        for budget in cfg.evaluation.budgets:
            metrics.update(evaluate(val_y, scores, budget))
        mlflow.log_metrics({f"val_{metric_key(k)}": v for k, v in metrics.items()})
        mlflow.log_metric("best_iteration", model.best_iteration_)
        mlflow.log_dict(categories, "categories.json")

        info = mlflow.lightgbm.log_model(model, name="model", registered_model_name=register_as)

        # Round trip: reload the logged model and rebuild features from the saved categories
        loaded = mlflow.lightgbm.load_model(info.model_uri)
        cats_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="categories.json"
        )
        saved_cats = json.loads(Path(cats_path).read_text())
        reloaded = predict_scores(loaded, apply_categories(val_x_raw, saved_cats))
        max_diff = float(np.abs(reloaded - scores).max())
        mlflow.log_metric("roundtrip_max_abs_diff", max_diff)

    print(f"run_id={run.info.run_id} git={sha[:7]} dirty={dirty} best_iter={model.best_iteration_}")
    for key, value in metrics.items():
        print(f"{key:>16s}: {value:.4f}")
    print(f"roundtrip_max_abs_diff: {max_diff:.2e}")
    name = cfg.mlflow.registered_model
    versions = mlflow.MlflowClient().search_model_versions(f"name='{name}'")
    print(f"registered {name} versions: {sorted(int(v.version) for v in versions)}")
    if max_diff > 1e-6:
        raise SystemExit("reloaded model does not reproduce the original scores")

    train_metrics = {f"val_{metric_key(k)}": round(v, 6) for k, v in metrics.items()}
    train_metrics["best_iteration"] = int(model.best_iteration_)
    train_metrics["roundtrip_max_abs_diff"] = max_diff
    TRAIN_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_METRICS_PATH.open("w") as f:
        json.dump(train_metrics, f, indent=2)


if __name__ == "__main__":
    main()
