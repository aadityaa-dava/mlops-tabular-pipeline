"""Train the raw-feature LightGBM and log it to MLflow: params, metrics, data version, model."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.artifacts
import mlflow.lightgbm
import numpy as np
import pandas as pd

from mlops_pipeline.gbm import TRAIN_END_DAY, align_categoricals, build_lgbm, split_xy
from mlops_pipeline.metrics import evaluate
from mlops_pipeline.split import time_split

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "fraud-ieee-cis"
MODEL_NAME = "fraud-lgbm"
VAL_END_DAY = 150.0
DVC_POINTERS = {
    "data_train_transaction_md5": "data/raw/train_transaction.csv.dvc",
    "data_train_identity_md5": "data/raw/train_identity.csv.dvc",
}


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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet("data/processed/train_merged.parquet")
    train, val, _test = time_split(df, TRAIN_END_DAY, VAL_END_DAY)  # test stays untouched
    train_x, train_y = split_xy(train)
    val_x_raw, val_y = split_xy(val)
    train_x, val_x = align_categoricals(train_x, val_x_raw)
    categories = fit_categories(train_x)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    sha, dirty = git_state()
    model = build_lgbm(args.seed)

    with mlflow.start_run(run_name=f"lgbm-raw-seed{args.seed}") as run:
        mlflow.set_tags({"git_commit": sha, "git_dirty": str(dirty), "scored_on": "validation"})
        mlflow.log_params(model.get_params())
        mlflow.log_params(
            {
                "seed": args.seed,
                "features": "raw",
                "train_end_day": TRAIN_END_DAY,
                "val_end_day": VAL_END_DAY,
                "train_rows": len(train),
                "val_rows": len(val),
                **{k: dvc_md5(p) for k, p in DVC_POINTERS.items()},
            }
        )

        model.fit(
            train_x,
            train_y,
            eval_X=val_x,
            eval_y=val_y,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        scores = predict_scores(model, val_x)
        metrics = {**evaluate(val_y, scores, 0.01), **evaluate(val_y, scores, 0.05)}
        mlflow.log_metrics({f"val_{metric_key(k)}": v for k, v in metrics.items()})
        mlflow.log_metric("best_iteration", model.best_iteration_)
        mlflow.log_dict(categories, "categories.json")

        info = mlflow.lightgbm.log_model(model, name="model", registered_model_name=MODEL_NAME)

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
    versions = mlflow.MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    print(f"registered {MODEL_NAME} versions: {sorted(int(v.version) for v in versions)}")
    if max_diff > 1e-6:
        raise SystemExit("reloaded model does not reproduce the original scores")


if __name__ == "__main__":
    main()
