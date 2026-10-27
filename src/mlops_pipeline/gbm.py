"""LightGBM experiments: raw columns vs engineered features, scored on validation."""

import argparse

import lightgbm as lgb
import pandas as pd

from mlops_pipeline.features import add_features
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


def build_lgbm(seed: int = 42) -> lgb.LGBMClassifier:
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
        random_state=seed,
        verbose=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=["raw", "batch1"], default="raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet("data/processed/train_merged.parquet")
    train, val, _test = time_split(df)  # test is still untouched
    if args.features == "batch1":
        train, val = add_features(train), add_features(val)

    train_x, train_y = split_xy(train)
    val_x, val_y = split_xy(val)
    train_x, val_x = align_categoricals(train_x, val_x)
    model = build_lgbm(args.seed).fit(
        train_x,
        train_y,
        eval_X=val_x,
        eval_y=val_y,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    scores = model.predict_proba(val_x)[:, 1]

    r1 = evaluate(val_y, scores, 0.01)
    r5 = evaluate(val_y, scores, 0.05)
    print(f"seed={args.seed} features={args.features} iters={model.best_iteration_}")
    print(f"pr_auc={r1['pr_auc']:.4f} roc_auc={r1['roc_auc']:.4f}")
    print(f"recall@1%={r1['recall_at_1%']:.4f} precision@1%={r1['precision_at_1%']:.4f}")
    print(f"recall@5%={r5['recall_at_5%']:.4f} precision@5%={r5['precision_at_5%']:.4f}")

    imp = pd.Series(model.booster_.feature_importance("gain"), index=train_x.columns)
    print("top 10 features by share of total gain:")
    print((imp / imp.sum()).sort_values(ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
