import pandas as pd
import pytest

from mlops_pipeline.split import check_no_leakage, time_split


def make_df() -> pd.DataFrame:
    days = [10, 50, 119.9, 120, 140, 149.9, 150, 170, 183]
    return pd.DataFrame(
        {
            "TransactionID": range(len(days)),
            "TransactionDT": [int(d * 86400) for d in days],
            "isFraud": [0, 1, 0, 0, 1, 0, 0, 0, 1],
        }
    )


def test_split_sizes_and_boundaries():
    train, val, test = time_split(make_df())
    assert (len(train), len(val), len(test)) == (3, 3, 3)
    assert 119.9 * 86400 in train["TransactionDT"].values
    assert 120 * 86400 in val["TransactionDT"].values  # boundary goes to val
    assert 150 * 86400 in test["TransactionDT"].values  # boundary goes to test


def test_splits_partition_all_rows_without_overlap():
    df = make_df()
    train, val, test = time_split(df)
    ids = list(train["TransactionID"]) + list(val["TransactionID"]) + list(test["TransactionID"])
    assert sorted(ids) == sorted(df["TransactionID"])


def test_invalid_cutoffs_raise():
    with pytest.raises(ValueError):
        time_split(make_df(), train_end_day=150, val_end_day=120)


def test_empty_split_raises():
    with pytest.raises(ValueError, match="empty"):
        time_split(make_df(), train_end_day=200, val_end_day=300)


def test_check_detects_shared_rows():
    df = make_df()
    with pytest.raises(ValueError):
        check_no_leakage(df, df, df)
