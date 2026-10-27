"""Baseline: logistic regression on a few simple features."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_pipeline.metrics import evaluate
from mlops_pipeline.split import time_split

C_COLS = [f"C{i}" for i in range(1, 15)]
CAT_COLS = ["ProductCD"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_amt"] = np.log1p(df["TransactionAmt"])
    hour = (df["TransactionDT"] // 3600) % 24  # raw TransactionDT is never a feature
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["has_identity"] = df["id_01"].notna().astype(int)
    for c in C_COLS:
        out[c] = np.log1p(df[c].clip(lower=0))
    out["ProductCD"] = df["ProductCD"].astype(str)
    return out


def build_model() -> Pipeline:
    num_cols = ["log_amt", "hour_sin", "hour_cos", "has_identity", *C_COLS]
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                num_cols,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])


def main() -> None:
    df = pd.read_parquet("data/processed/train_merged.parquet")
    train, val, _test = time_split(df)  # test is deliberately not used here
    model = build_model().fit(make_features(train), train["isFraud"])
    scores = model.predict_proba(make_features(val))[:, 1]
    print(f"validation fraud rate (random-guess PR-AUC): {val['isFraud'].mean():.4f}")
    for name, value in evaluate(val["isFraud"], scores).items():
        print(f"{name:>16s}: {value:.4f}")


if __name__ == "__main__":
    main()
