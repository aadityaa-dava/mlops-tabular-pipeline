"""Serving logic without MLflow: validate requests and rebuild the model's input frame."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlops_pipeline.tracking import apply_categories

MAX_BATCH = 1000


class Transaction(BaseModel):
    """One transaction.

    TransactionAmt and ProductCD are required. Any other column the model uses is optional;
    a missing value is NaN, exactly as in training. Columns the model does not use are
    rejected later, in build_frame, because only the loaded model knows its feature list.
    """

    model_config = ConfigDict(extra="allow")
    TransactionAmt: float = Field(gt=0, allow_inf_nan=False)
    ProductCD: Literal["W", "C", "R", "H", "S"]


class PredictRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH)


def records_from_request(request: PredictRequest) -> list[dict[str, Any]]:
    return [t.model_dump() for t in request.transactions]


def build_frame(
    records: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    categories: Mapping[str, list[str]],
) -> pd.DataFrame:
    """The model's input frame: training column order, float32 numerics, training categories."""
    names = list(feature_names)
    unknown = sorted({key for r in records for key in r} - set(names))
    if unknown:
        raise ValueError(f"columns the model does not use: {unknown[:10]}")
    ghost = sorted(set(categories) - set(names))
    if ghost:
        raise ValueError(f"categories given for columns the model does not use: {ghost[:10]}")
    df = pd.DataFrame(list(records)).reindex(columns=names)
    for col in names:
        if col not in categories:
            df[col] = pd.to_numeric(df[col], errors="raise").astype("float32")
    return apply_categories(df, categories)
