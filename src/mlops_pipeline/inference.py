"""Scoring helpers with no MLflow dependency, shared by training, export and serving."""

import numpy as np
import pandas as pd


def apply_categories(x: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Cast text columns to the training categories; unseen or rare values become NaN."""
    out = x.copy()
    for col, cats in categories.items():
        out[col] = out[col].where(out[col].isin(cats)).astype(pd.CategoricalDtype(cats))
    return out


def predict_scores(model, x: pd.DataFrame) -> np.ndarray:
    """Fraud probabilities from a scikit-learn style model or a raw LightGBM Booster."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)
