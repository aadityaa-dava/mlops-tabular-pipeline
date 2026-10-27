"""Time-based train/validation/test split with explicit leakage checks."""

import pandas as pd

SECONDS_PER_DAY = 86_400


def time_split(
    df: pd.DataFrame,
    train_end_day: float = 120.0,
    val_end_day: float = 150.0,
    time_col: str = "TransactionDT",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: train < train_end_day <= val < val_end_day <= test."""
    if not 0 < train_end_day < val_end_day:
        raise ValueError("need 0 < train_end_day < val_end_day")
    day = df[time_col] / SECONDS_PER_DAY
    train = df[day < train_end_day]
    val = df[(day >= train_end_day) & (day < val_end_day)]
    test = df[day >= val_end_day]
    check_no_leakage(train, val, test, time_col)
    return train, val, test


def check_no_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    time_col: str = "TransactionDT",
    id_col: str = "TransactionID",
) -> None:
    """Raise ValueError if splits are empty, overlap in time, or share rows."""
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part.empty:
            raise ValueError(f"{name} split is empty")
    if not train[time_col].max() < val[time_col].min():
        raise ValueError("train overlaps val in time")
    if not val[time_col].max() < test[time_col].min():
        raise ValueError("val overlaps test in time")
    ids = [set(p[id_col]) for p in (train, val, test)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("TransactionID appears in more than one split")
