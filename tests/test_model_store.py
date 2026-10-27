import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from mlops_pipeline.inference import apply_categories, predict_scores
from mlops_pipeline.model_store import REQUIRED_FILES, load_exported, write_export

CATEGORIES = {"c": ["a", "b"]}


def fit_tiny_model():
    rng = np.random.default_rng(0)
    n = 400
    raw = pd.DataFrame(
        {"num": rng.normal(size=n).astype("float32"), "c": rng.choice(["a", "b", "z"], n)}
    )
    y = ((raw["num"] > 0) | (raw["c"] == "a")).astype(int)
    x = apply_categories(raw, CATEGORIES)
    model = lgb.LGBMClassifier(
        n_estimators=15, num_leaves=4, min_child_samples=5, random_state=0, verbose=-1
    ).fit(x, y)
    return model, x


def test_exported_model_scores_like_the_original(tmp_path):
    model, x = fit_tiny_model()
    write_export(tmp_path, model.booster_, CATEGORIES, {"model_version": 3})
    loaded = load_exported(tmp_path)
    assert loaded.version == "3"
    assert loaded.feature_names == ["num", "c"]
    assert loaded.categories == CATEGORIES
    np.testing.assert_allclose(
        predict_scores(loaded.model, x), predict_scores(model, x), rtol=0, atol=1e-12
    )


def test_missing_folder_points_to_make_export(tmp_path):
    with pytest.raises(FileNotFoundError, match="make export"):
        load_exported(tmp_path / "nope")


@pytest.mark.parametrize("missing", REQUIRED_FILES)
def test_incomplete_export_is_rejected(tmp_path, missing):
    model, _ = fit_tiny_model()
    write_export(tmp_path, model.booster_, CATEGORIES, {"model_version": 1})
    (tmp_path / missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        load_exported(tmp_path)


def test_serving_modules_do_not_import_mlflow():
    code = "import sys, mlops_pipeline.api; print('mlflow' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
