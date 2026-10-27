"""Typed project configuration, loaded from YAML. Unknown or invalid keys fail loudly."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    raw_dir: Path
    merged_path: Path


class SplitConfig(_Strict):
    train_end_day: float
    val_end_day: float

    @model_validator(mode="after")
    def check_order(self) -> "SplitConfig":
        if not 0 < self.train_end_day < self.val_end_day:
            raise ValueError("need 0 < train_end_day < val_end_day")
        return self


class ModelConfig(_Strict):
    seed: int
    early_stopping_rounds: int
    min_category_count: int
    params: dict[str, float | int | str]


class EvaluationConfig(_Strict):
    budgets: list[float]

    @field_validator("budgets")
    @classmethod
    def check_budgets(cls, budgets: list[float]) -> list[float]:
        if not budgets or any(not 0 < b <= 1 for b in budgets):
            raise ValueError("budgets must be a non-empty list of fractions in (0, 1]")
        return budgets


class MlflowConfig(_Strict):
    tracking_uri: str
    experiment: str
    registered_model: str


class Config(_Strict):
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    mlflow: MlflowConfig


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    return Config.model_validate(yaml.safe_load(Path(path).read_text()))
