"""Self-contained model export: write a folder, load it back without MLflow or training code."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb

DEFAULT_MODEL_DIR = "artifacts/champion"
MODEL_FILE = "model.txt"
CATEGORIES_FILE = "categories.json"
METADATA_FILE = "metadata.json"
REQUIRED_FILES = (MODEL_FILE, CATEGORIES_FILE, METADATA_FILE)


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    feature_names: list[str]
    categories: dict[str, list[str]]
    version: str


def write_export(
    out_dir: Path | str,
    booster: lgb.Booster,
    categories: dict[str, list[str]],
    metadata: dict[str, Any],
    num_iteration: int | None = None,
) -> None:
    """Write the model, its category lists and metadata into one folder."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / MODEL_FILE), num_iteration=num_iteration)
    (out / CATEGORIES_FILE).write_text(json.dumps(categories, indent=2) + "\n")
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_exported(model_dir: Path | str = DEFAULT_MODEL_DIR) -> LoadedModel:
    """Load an exported folder. Needs LightGBM only: no MLflow, no config, no data."""
    folder = Path(model_dir)
    missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{folder} is missing {missing}; run `make export` first")
    booster = lgb.Booster(model_file=str(folder / MODEL_FILE))
    categories = json.loads((folder / CATEGORIES_FILE).read_text())
    metadata = json.loads((folder / METADATA_FILE).read_text())
    return LoadedModel(
        model=booster,
        feature_names=list(booster.feature_name()),
        categories=categories,
        version=str(metadata["model_version"]),
    )
