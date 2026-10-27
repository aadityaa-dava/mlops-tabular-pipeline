from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mlops_pipeline.config import Config, load_config
from mlops_pipeline.gbm import build_lgbm

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def raw_dict() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_repo_config_loads_and_drives_the_model():
    cfg = load_config(CONFIG_PATH)
    params = build_lgbm(cfg.model).get_params()
    assert params["random_state"] == cfg.model.seed
    for key, value in cfg.model.params.items():
        assert params[key] == value, key


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
