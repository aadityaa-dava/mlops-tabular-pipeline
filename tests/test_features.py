import numpy as np
import pandas as pd

from mlops_pipeline.features import add_features


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionDT": [3600 * 5, 86400 * 8, 86400 * 8 + 3600],
            "TransactionAmt": [10.5, 20.0, 7.25],
            "P_emaildomain": ["gmail.com", "yahoo.com", None],
            "R_emaildomain": ["gmail.com", "gmail.com", "gmail.com"],
            "D1": [1.0, np.nan, np.nan],
            "D2": [np.nan, np.nan, np.nan],
            "M1": ["T", None, None],
            "V1": [1.0, 2.0, np.nan],
            "V2": [np.nan, 2.0, np.nan],
            "id_01": [np.nan, -5.0, np.nan],
            "id_02": [np.nan, 100.0, np.nan],
        }
    )


def test_time_and_amount_features():
    out = add_features(make_df())
    assert out["hour"].tolist() == [5, 0, 1]
    assert out["dow"].tolist() == [0, 1, 1]
    assert out["amt_frac"].tolist() == [0.5, 0.0, 0.25]


def test_missing_counts_and_identity_flag():
    out = add_features(make_df())
    assert out["nan_D"].tolist() == [1, 2, 2]
    assert out["nan_M"].tolist() == [0, 1, 1]
    assert out["nan_V"].tolist() == [1, 0, 2]
    assert out["nan_id"].tolist() == [2, 0, 2]
    assert out["has_identity"].tolist() == [0, 1, 0]


def test_email_match_is_three_valued():
    out = add_features(make_df())
    assert out["email_match"].iloc[:2].tolist() == [1.0, 0.0]
    assert np.isnan(out["email_match"].iloc[2])  # unknown stays unknown


def test_input_not_mutated_and_no_label_needed():
    df = make_df()
    assert "isFraud" not in df.columns  # serving requests have no label
    before = df.copy()
    add_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_row_wise_so_no_cross_split_leakage():
    df = make_df()
    pd.testing.assert_frame_equal(add_features(df.iloc[:2]), add_features(df).iloc[:2])
