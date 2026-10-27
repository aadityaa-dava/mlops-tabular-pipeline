"""Train/serve skew check on real rows.

Scores from the deployed path (exported folder, JSON over HTTP) must match scores from the
training-side path (model loaded from the MLflow registry, batch scoring).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_pipeline.api import create_app
from mlops_pipeline.config import load_config
from mlops_pipeline.export import fetch_champion
from mlops_pipeline.gbm import split_xy
from mlops_pipeline.inference import apply_categories, predict_scores
from mlops_pipeline.model_store import DEFAULT_MODEL_DIR, load_exported
from mlops_pipeline.serving import MAX_BATCH
from mlops_pipeline.split import time_split

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "config.yaml")
DB = Path(CFG.mlflow.tracking_uri.removeprefix("sqlite:///"))
EXPORT_DIR = Path(DEFAULT_MODEL_DIR)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (Path(CFG.data.merged_path).exists() and DB.exists() and EXPORT_DIR.is_dir()),
        reason="needs the merged parquet, mlflow.db and an exported model (make export)",
    ),
]


def to_records(x: pd.DataFrame) -> list[dict]:
    """JSON-ready records: missing values are left out, as a real client would."""
    return [
        {key: value for key, value in row.items() if pd.notna(value)}
        for row in x.to_dict(orient="records")
    ]


def test_exported_http_scores_match_registry_batch_scores_on_real_rows():
    registry_model, categories, metadata = fetch_champion(CFG)
    exported = load_exported(EXPORT_DIR)
    stale = "export is stale: run `make export`"
    assert exported.version == str(metadata["model_version"]), stale
    assert exported.categories == categories

    df = pd.read_parquet(CFG.data.merged_path)
    _train, val, _test = time_split(df, CFG.split.train_end_day, CFG.split.val_end_day)
    x, _y = split_xy(val.sample(2000, random_state=0))
    batch = predict_scores(registry_model, apply_categories(x, categories))

    records = to_records(x)
    served: list[float] = []
    with TestClient(create_app(lambda: exported)) as client:
        for start in range(0, len(records), MAX_BATCH):
            chunk = records[start : start + MAX_BATCH]
            resp = client.post("/predict", json={"transactions": chunk})
            assert resp.status_code == 200, resp.text
            served.extend(resp.json()["scores"])

    assert len(served) == len(batch)
    max_diff = float(np.abs(np.array(served) - batch).max())
    print(f"rows={len(served)} max_abs_diff={max_diff:.3e}")
    assert max_diff < 1e-9, f"serving differs from batch scoring by {max_diff:.3e}"
