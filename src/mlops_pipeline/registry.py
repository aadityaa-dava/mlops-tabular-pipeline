"""Apply the promotion policy to a registered model version and score the test window once."""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.lightgbm
import pandas as pd
from mlflow import MlflowClient

from mlops_pipeline.config import load_config
from mlops_pipeline.gbm import split_xy
from mlops_pipeline.metrics import evaluate
from mlops_pipeline.promotion import should_promote
from mlops_pipeline.split import time_split
from mlops_pipeline.tracking import apply_categories, metric_key, predict_scores


def promotion_metrics(run_metrics: dict[str, float]) -> dict[str, float]:
    """Map logged validation metric names to the keys the promotion rule expects.

    A missing round-trip check counts as a failure, so unverified models are never promoted.
    """
    out = {k.removeprefix("val_"): v for k, v in run_metrics.items() if k.startswith("val_")}
    out["roundtrip_max_abs_diff"] = run_metrics.get("roundtrip_max_abs_diff", float("inf"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--version", type=int, default=None, help="default: latest version")
    args = parser.parse_args()
    cfg = load_config(args.config)
    name = cfg.mlflow.registered_model

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    version = args.version or max(int(v.version) for v in versions)
    mv = client.get_model_version(name, str(version))
    if not mv.run_id:
        raise SystemExit(f"model version {version} has no linked run")
    run = client.get_run(mv.run_id)

    exp = client.get_experiment_by_name(cfg.mlflow.experiment)
    baseline_run = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.mlflow.runName = 'baseline'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )[0]

    decision = should_promote(
        promotion_metrics(run.data.metrics),
        baseline=promotion_metrics(baseline_run.data.metrics),
    )
    print(f"{name} v{version}: promote={decision.promote} ({decision.reason})")
    if not decision.promote:
        raise SystemExit("not promoted; test window not scored")

    client.set_registered_model_alias(name, "champion", str(version))
    champion = client.get_model_version_by_alias(name, "champion")
    print(f"alias 'champion' now points to v{champion.version}")

    if "test_pr_auc" in run.data.metrics:
        raise SystemExit("test window already scored for this run; refusing to score it again")

    df = pd.read_parquet(cfg.data.merged_path)
    _train, _val, test = time_split(df, cfg.split.train_end_day, cfg.split.val_end_day)
    x, y = split_xy(test)
    cats_path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="categories.json"
    )
    categories = json.loads(Path(cats_path).read_text())
    model = mlflow.lightgbm.load_model(f"models:/{name}/{version}")
    scores = predict_scores(model, apply_categories(x, categories))
    test_metrics: dict[str, float] = {}
    for budget in cfg.evaluation.budgets:
        test_metrics.update(evaluate(y, scores, budget))

    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metrics({f"test_{metric_key(k)}": v for k, v in test_metrics.items()})
        mlflow.log_metric("test_rows", len(test))
        mlflow.set_tag("test_scored", "yes")

    print(f"\ntest window: {len(test)} rows, fraud rate {y.mean():.4f}")
    print(f"{'metric':>16s} {'validation':>11s} {'test':>8s}")
    for key, value in test_metrics.items():
        val_value = run.data.metrics[f"val_{metric_key(key)}"]
        print(f"{key:>16s} {val_value:11.4f} {value:8.4f}")


if __name__ == "__main__":
    main()
