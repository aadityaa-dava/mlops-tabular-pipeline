import inspect
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mlops_pipeline.config import Config, load_config
from mlops_pipeline.gbm import TRAIN_END_DAY, align_categoricals, build_lgbm
from mlops_pipeline.tracking import EXPERIMENT, MODEL_NAME, TRACKING_URI, VAL_END_DAY

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def raw_dict() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_config_reproduces_current_hard_coded_values():
    cfg = load_config(CONFIG_PATH)
    assert cfg.split.train_end_day == TRAIN_END_DAY
    assert cfg.split.val_end_day == VAL_END_DAY
    assert cfg.mlflow.tracking_uri == TRACKING_URI
    assert cfg.mlflow.experiment == EXPERIMENT
    assert cfg.mlflow.registered_model == MODEL_NAME
    default_min = inspect.signature(align_categoricals).parameters["min_count"].default
    assert cfg.model.min_category_count == default_min
    model_params = build_lgbm(cfg.model.seed).get_params()
    assert model_params["random_state"] == cfg.model.seed
    for key, value in cfg.model.params.items():
        assert model_params[key] == value, key


def test_rejects_unordered_cutoffs():
    d = raw_dict()
    d["split"]["train_end_day"] = 200.0
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_rejects_unknown_keys():
    d = raw_dict()
    d["split"]["trian_end_day"] = 100.0  # typo must not be silently ignored
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_rejects_bad_budget():
    d = raw_dict()
    d["evaluation"]["budgets"] = [0.0, 1.5]
    with pytest.raises(ValidationError):
        Config.model_validate(d)
