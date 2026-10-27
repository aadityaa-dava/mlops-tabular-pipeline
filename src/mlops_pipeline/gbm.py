"""LightGBM helpers, plus a CLI that compares raw columns with engineered features."""

import argparse

import lightgbm as lgb
import pandas as pd

from mlops_pipeline.config import ModelConfig, load_config
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


def align_categoricals(train_x: pd.DataFrame, *others: pd.DataFrame, min_count: int):
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


def trim_train(
    train: pd.DataFrame,
    gap_days: float,
    trim_early_days: float,
    train_end_day: float,
) -> pd.DataFrame:
    """Drop the last gap_days (a gap before validation) and/or the first trim_early_days."""
    day = train["TransactionDT"] / 86_400
    return train[(day < train_end_day - gap_days) & (day >= trim_early_days)]


def build_lgbm(model_cfg: ModelConfig, seed: int | None = None) -> lgb.LGBMClassifier:
    """LightGBM classifier from the config. A given seed overrides the configured one."""
    return lgb.LGBMClassifier(
        **model_cfg.params,
        random_state=model_cfg.seed if seed is None else seed,
        verbose=-1,
    )


def fit_lgbm(model, train_x, train_y, val_x, val_y, early_stopping_rounds: int):
    """Fit with early stopping on the validation window and return the fitted model."""
    return model.fit(
        train_x,
        train_y,
        eval_X=val_x,
        eval_y=val_y,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--features", choices=["raw", "batch1"], default="raw")
    parser.add_argument("--seed", type=int, default=None, help="default: the config's seed")
    parser.add_argument("--gap-days", type=float, default=0.0)
    parser.add_argument("--trim-early-days", type=float, default=0.0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg.model.seed if args.seed is None else args.seed

    df = pd.read_parquet(cfg.data.merged_path)
    # the test window is never used here
    train, val, _test = time_split(df, cfg.split.train_end_day, cfg.split.val_end_day)
    train = trim_train(train, args.gap_days, args.trim_early_days, cfg.split.train_end_day)
    if args.features == "batch1":
        train, val = add_features(train), add_features(val)

    train_x, train_y = split_xy(train)
    val_x, val_y = split_xy(val)
    train_x, val_x = align_categoricals(train_x, val_x, min_count=cfg.model.min_category_count)
    model = fit_lgbm(
        build_lgbm(cfg.model, seed),
        train_x,
        train_y,
        val_x,
        val_y,
        cfg.model.early_stopping_rounds,
    )
    scores = model.predict_proba(val_x)[:, 1]

    print(f"seed={seed} features={args.features} iters={model.best_iteration_}")
    print(f"gap_days={args.gap_days} trim_early_days={args.trim_early_days} rows={len(train)}")
    first = evaluate(val_y, scores, cfg.evaluation.budgets[0])
    print(f"pr_auc={first['pr_auc']:.4f} roc_auc={first['roc_auc']:.4f}")
    for budget in cfg.evaluation.budgets:
        tag = f"{budget:.0%}"
        m = evaluate(val_y, scores, budget)
        recall, precision = m[f"recall_at_{tag}"], m[f"precision_at_{tag}"]
        print(f"recall@{tag}={recall:.4f} precision@{tag}={precision:.4f}")

    imp = pd.Series(model.booster_.feature_importance("gain"), index=train_x.columns)
    print("top 10 features by share of total gain:")
    print((imp / imp.sum()).sort_values(ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
