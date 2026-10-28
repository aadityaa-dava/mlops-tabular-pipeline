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
IMPORTANCE_FILE = "feature_importance.json"  # optional; read by monitoring only
REQUIRED_FILES = (MODEL_FILE, CATEGORIES_FILE, METADATA_FILE)


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    feature_names: list[str]
    categories: dict[str, list[str]]
    version: str


def gain_shares(booster: lgb.Booster) -> dict[str, float]:
    """Each feature's share of the total split gain, largest first."""
    gains = booster.feature_importance(importance_type="gain")
    total = float(gains.sum())
    if total <= 0:
        return {}
    names = booster.feature_name()
    shares = {name: float(g) / total for name, g in zip(names, gains, strict=True)}
    return dict(sorted(shares.items(), key=lambda item: -item[1]))


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
    saved = lgb.Booster(model_file=str(out / MODEL_FILE))  # importance of what is served
    (out / IMPORTANCE_FILE).write_text(json.dumps(gain_shares(saved), indent=2) + "\n")


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
