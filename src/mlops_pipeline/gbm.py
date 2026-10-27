"""LightGBM on the raw columns (no feature engineering yet), compared to the baseline."""

import lightgbm as lgb
import pandas as pd

from mlops_pipeline.baseline import build_model as build_baseline
from mlops_pipeline.baseline import make_features
from mlops_pipeline.metrics import evaluate
from mlops_pipeline.split import time_split

DROP_COLS = ["TransactionID", "TransactionDT", "isFraud"]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features and label. ID and raw time are never features (see eda_findings.md)."""
    return df.drop(columns=DROP_COLS), df["isFraud"]


def _to_categorical(values: pd.Series, dtype: pd.CategoricalDtype) -> pd.Series:
    """Cast to dtype, explicitly mapping values outside its categories to NaN."""
    return values.where(values.isin(dtype.categories)).astype(dtype)


def align_categoricals(train_x: pd.DataFrame, *others: pd.DataFrame, min_count: int = 100):
    """Turn text columns into categoricals using categories seen >= min_count times in train.

    Rare or unseen values become NaN, identically in every split, so there is no
    leakage from val/test and no category mismatch between training and serving.
    """
    cat_cols = list(train_x.select_dtypes(exclude="number").columns)
    train_x = train_x.copy()
    others = tuple(o.copy() for o in others)
    for col in cat_cols:
        counts = train_x[col].value_counts()
        dtype = pd.CategoricalDtype(counts[counts >= min_count].index.tolist())
        train_x[col] = _to_categorical(train_x[col], dtype)
        for o in others:
            o[col] = _to_categorical(o[col], dtype)
    return (train_x, *others)


def build_lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        metric="average_precision",
        random_state=42,
        verbose=-1,
    )


def main() -> None:
    df = pd.read_parquet("data/processed/train_merged.parquet")
    train, val, _test = time_split(df)  # test is still untouched

    base = build_baseline().fit(make_features(train), train["isFraud"])
    base_scores = base.predict_proba(make_features(val))[:, 1]

    train_x, train_y = split_xy(train)
    val_x, val_y = split_xy(val)
    train_x, val_x = align_categoricals(train_x, val_x)
    model = build_lgbm().fit(
        train_x,
        train_y,
        eval_X=val_x,
        eval_y=val_y,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    scores = model.predict_proba(val_x)[:, 1]

    print(f"best iteration: {model.best_iteration_}  (features: {train_x.shape[1]})")
    headers = ["pr_auc", "roc_auc", "rec@1%", "prec@1%", "rec@5%", "prec@5%"]
    print(f"{'model':10s} " + " ".join(f"{h:>8s}" for h in headers))
    for name, s in (("baseline", base_scores), ("lightgbm", scores)):
        r1 = evaluate(val["isFraud"], s, 0.01)
        r5 = evaluate(val["isFraud"], s, 0.05)
        row = [
            r1["pr_auc"],
            r1["roc_auc"],
            r1["recall_at_1%"],
            r1["precision_at_1%"],
            r5["recall_at_5%"],
            r5["precision_at_5%"],
        ]
        print(f"{name:10s} " + " ".join(f"{v:8.4f}" for v in row))

    imp = pd.Series(model.booster_.feature_importance("gain"), index=train_x.columns)
    print("\ntop 10 features by share of total gain:")
    print((imp / imp.sum()).sort_values(ascending=False).head(10).round(3))


if __name__ == "__main__":
    main()
