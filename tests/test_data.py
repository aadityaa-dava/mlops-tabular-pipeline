import pandas as pd
import pytest
from pandas.errors import MergeError

from mlops_pipeline.data import load_merged


def write_raw(tmp_path, identity_ids=(1, 3)):
    pd.DataFrame(
        {
            "TransactionID": [3, 1, 2],
            "TransactionDT": [300, 100, 200],
            "isFraud": [0, 1, 0],
            "TransactionAmt": [5.5, 6.5, 7.5],
        }
    ).to_csv(tmp_path / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": list(identity_ids), "id_01": [-5.0] * len(identity_ids)}).to_csv(
        tmp_path / "train_identity.csv", index=False
    )


def test_left_join_keeps_all_transactions_and_sorts_by_time(tmp_path):
    write_raw(tmp_path)
    df = load_merged(tmp_path)
    assert df["TransactionID"].tolist() == [1, 2, 3]
    assert df["TransactionDT"].is_monotonic_increasing
    assert df["id_01"].isna().tolist() == [False, True, False]  # transaction 2 has no identity


def test_float_columns_are_downcast_to_float32(tmp_path):
    write_raw(tmp_path)
    df = load_merged(tmp_path)
    assert df["TransactionAmt"].dtype == "float32"
    assert df["id_01"].dtype == "float32"


def test_duplicate_identity_ids_raise(tmp_path):
    write_raw(tmp_path, identity_ids=(1, 1, 3))
    with pytest.raises(MergeError):
        load_merged(tmp_path)
