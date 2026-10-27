import pandas as pd
import pandera.pandas as pa
import pytest

from mlops_pipeline.schema import validate_transactions


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionDT": [100, 200, 300],
            "TransactionAmt": [10.5, 99.0, 5.25],
            "ProductCD": ["W", "C", "R"],
            "card4": ["visa", None, "mastercard"],
        }
    )


def test_valid_frame_passes():
    validate_transactions(make_df())


def test_bad_label_fails():
    df = make_df()
    df.loc[0, "isFraud"] = 2
    with pytest.raises(pa.errors.SchemaErrors):
        validate_transactions(df)


def test_duplicate_id_fails():
    df = make_df()
    df.loc[1, "TransactionID"] = 1
    with pytest.raises(pa.errors.SchemaErrors):
        validate_transactions(df)


def test_non_positive_amount_fails():
    df = make_df()
    df.loc[2, "TransactionAmt"] = -1.0
    with pytest.raises(pa.errors.SchemaErrors):
        validate_transactions(df)


def test_unsorted_time_fails():
    df = make_df()
    df["TransactionDT"] = [300, 200, 100]
    with pytest.raises(pa.errors.SchemaErrors):
        validate_transactions(df)
