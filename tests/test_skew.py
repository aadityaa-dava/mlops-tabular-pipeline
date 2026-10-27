"""Train/serve skew check: the HTTP path must score real rows exactly like the batch path."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_pipeline.api import create_app, load_champion
from mlops_pipeline.config import load_config
from mlops_pipeline.gbm import split_xy
from mlops_pipeline.serving import MAX_BATCH
from mlops_pipeline.split import time_split
from mlops_pipeline.tracking import apply_categories, predict_scores

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "config.yaml")
DB = Path(CFG.mlflow.tracking_uri.removeprefix("sqlite:///"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (Path(CFG.data.merged_path).exists() and DB.exists()),
        reason="needs the merged parquet and mlflow.db (run from the project root)",
    ),
]


def to_records(x: pd.DataFrame) -> list[dict]:
    """JSON-ready records: missing values are left out, as a real client would."""
    return [
        {key: value for key, value in row.items() if pd.notna(value)}
        for row in x.to_dict(orient="records")
    ]


def test_http_scores_match_batch_scores_on_real_validation_rows():
    champion = load_champion(CFG)
    df = pd.read_parquet(CFG.data.merged_path)
    _train, val, _test = time_split(df, CFG.split.train_end_day, CFG.split.val_end_day)
    x, _y = split_xy(val.sample(2000, random_state=0))
    batch = predict_scores(champion.model, apply_categories(x, champion.categories))

    records = to_records(x)
    served: list[float] = []
    with TestClient(create_app(lambda: champion)) as client:
        for start in range(0, len(records), MAX_BATCH):
            chunk = records[start : start + MAX_BATCH]
            resp = client.post("/predict", json={"transactions": chunk})
            assert resp.status_code == 200, resp.text
            served.extend(resp.json()["scores"])

    assert len(served) == len(batch)
    max_diff = float(np.abs(np.array(served) - batch).max())
    print(f"rows={len(served)} max_abs_diff={max_diff:.3e}")
    assert max_diff < 1e-9, f"serving differs from batch scoring by {max_diff:.3e}"
