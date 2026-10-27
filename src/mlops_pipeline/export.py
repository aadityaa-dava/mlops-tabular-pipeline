"""Export the registry champion to a self-contained folder: the unit that gets deployed."""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.lightgbm
from mlflow import MlflowClient

from mlops_pipeline.config import Config, load_config
from mlops_pipeline.model_store import DEFAULT_MODEL_DIR, write_export
from mlops_pipeline.registry import CHAMPION_ALIAS


def fetch_champion(cfg: Config):
    """The champion from the registry: (fitted model, category lists, metadata)."""
    name = cfg.mlflow.registered_model
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    run = client.get_run(mv.run_id)
    model = mlflow.lightgbm.load_model(f"models:/{name}/{mv.version}")
    cats_path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="categories.json"
    )
    categories = json.loads(Path(cats_path).read_text())
    metadata = {
        "registered_model": name,
        "model_version": int(mv.version),
        "run_id": mv.run_id,
        "git_commit": run.data.tags.get("git_commit"),
        "data_train_transaction_md5": run.data.params.get("data_train_transaction_md5"),
        "data_train_identity_md5": run.data.params.get("data_train_identity_md5"),
        "val_pr_auc": run.data.metrics.get("val_pr_auc"),
        "best_iteration": model.best_iteration_,
    }
    return model, categories, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    model, categories, metadata = fetch_champion(load_config(args.config))
    write_export(
        args.out,
        model.booster_,
        categories,
        metadata,
        num_iteration=model.best_iteration_,
    )
    print(f"exported {metadata['registered_model']} v{metadata['model_version']} to {args.out}")


if __name__ == "__main__":
    main()
