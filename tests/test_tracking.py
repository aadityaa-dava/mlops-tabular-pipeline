import pandas as pd
import pytest

from mlops_pipeline.gbm import align_categoricals
from mlops_pipeline.tracking import apply_categories, dvc_md5, fit_categories, metric_key


def test_metric_key_removes_percent():
    assert metric_key("recall_at_1%") == "recall_at_1pct"


def test_dvc_md5_parses_pointer_file(tmp_path):
    f = tmp_path / "x.csv.dvc"
    f.write_text("outs:\n- md5: 8487db5001c8bad139f3318d5d3db416\n  hash: md5\n  path: x.csv\n")
    assert dvc_md5(str(f)) == "8487db5001c8bad139f3318d5d3db416"


def test_dvc_md5_raises_when_missing(tmp_path):
    f = tmp_path / "bad.dvc"
    f.write_text("outs: []\n")
    with pytest.raises(ValueError):
        dvc_md5(str(f))


def make_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame({"c": ["a"] * 150 + ["b"] * 120 + ["rare"] * 3, "n": range(273)})
    val = pd.DataFrame({"c": ["a", "b", "rare", "zzz"], "n": [1, 2, 3, 4]})
    return train, val


def test_saved_categories_reproduce_training_alignment():
    train, val = make_frames()
    aligned_train, aligned_val = align_categoricals(train, val)
    cats = fit_categories(aligned_train)
    assert cats == {"c": ["a", "b"]}  # numeric column ignored, rare category dropped
    pd.testing.assert_frame_equal(apply_categories(val, cats), aligned_val)


def test_apply_categories_does_not_mutate_input():
    _, val = make_frames()
    before = val.copy()
    apply_categories(val, {"c": ["a", "b"]})
    pd.testing.assert_frame_equal(val, before)
