"""Apply the promotion policy to a registered model version; score the test window once ever."""

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
from mlops_pipeline.promotion import should_promote, window_mismatch
from mlops_pipeline.split import time_split
from mlops_pipeline.tracking import apply_categories, metric_key, predict_scores

CHAMPION_ALIAS = "champion"


def promotion_metrics(run_metrics: dict[str, float]) -> dict[str, float]:
    """Map logged validation metric names to the keys the promotion rule expects.

    A missing round-trip check counts as a failure, so unverified models are never promoted.
    """
    out = {k.removeprefix("val_"): v for k, v in run_metrics.items() if k.startswith("val_")}
    out["roundtrip_max_abs_diff"] = run_metrics.get("roundtrip_max_abs_diff", float("inf"))
    return out


def get_champion(client: MlflowClient, name: str):
    """The model version holding the champion alias, or None if there is none yet."""
    models = client.search_registered_models(filter_string=f"name='{name}'")
    if not models:
        return None
    version = models[0].aliases.get(CHAMPION_ALIAS)
    return client.get_model_version(name, str(version)) if version is not None else None


def find_test_scoring_run(client: MlflowClient, experiment_id: str) -> str | None:
    """Run id of a run that already scored the test window, or None.

    After one run has scored it, the window is spent for model selection.
    """
    runs = client.search_runs(
        [experiment_id], filter_string="tags.test_scored = 'yes'", max_results=1
    )
    return runs[0].info.run_id if runs else None


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

    champion = get_champion(client, name)
    if champion is not None and int(champion.version) == version:
        print(f"{name} v{version} is already the champion; nothing to do")
        return

    exp = client.get_experiment_by_name(cfg.mlflow.experiment)
    if champion is not None:
        reference_run = client.get_run(champion.run_id)
        label = f"champion v{champion.version}"
    else:
        reference_run = client.search_runs(
            [exp.experiment_id],
            filter_string="tags.mlflow.runName = 'baseline'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )[0]
        label = "baseline"
    mismatch = window_mismatch(run.data.params, reference_run.data.params)
    if mismatch:
        raise SystemExit(f"not comparable with the {label}: {mismatch}")

    candidate = promotion_metrics(run.data.metrics)
    reference = promotion_metrics(reference_run.data.metrics)
    if champion is not None:
        decision = should_promote(candidate, champion=reference)
    else:
        decision = should_promote(candidate, baseline=reference)
    print(f"{name} v{version} vs {label}: promote={decision.promote} ({decision.reason})")
    if not decision.promote:
        print("not promoted: the champion is unchanged and the test window is untouched")
        return

    client.set_registered_model_alias(name, CHAMPION_ALIAS, str(version))
    print(f"alias '{CHAMPION_ALIAS}' now points to v{version}")

    spent_by = find_test_scoring_run(client, exp.experiment_id)
    if spent_by is not None:
        print(f"test window already used by run {spent_by}; not scoring it again")
        return

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
