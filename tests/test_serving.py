import numpy as np
import pytest
from pydantic import ValidationError

from mlops_pipeline.serving import (
    MAX_BATCH,
    PredictRequest,
    build_frame,
    records_from_request,
)

FEATURES = ["TransactionAmt", "card1", "ProductCD", "P_emaildomain", "id_01"]
CATEGORIES = {"ProductCD": ["W", "C"], "P_emaildomain": ["gmail.com"]}


def rec(**extra):
    return {"TransactionAmt": 10.5, "ProductCD": "W", **extra}


def test_frame_has_training_columns_in_order_with_training_types():
    df = build_frame([rec(card1=123, P_emaildomain="gmail.com")], FEATURES, CATEGORIES)
    assert list(df.columns) == FEATURES
    assert df["TransactionAmt"].dtype == np.float32
    assert df["card1"].dtype == np.float32
    assert list(df["ProductCD"].cat.categories) == ["W", "C"]
    assert df["ProductCD"].iloc[0] == "W"


def test_rare_or_unseen_category_becomes_nan():
    records = [rec(P_emaildomain="weird.example"), rec(ProductCD="H")]
    df = build_frame(records, FEATURES, CATEGORIES)
    assert df["P_emaildomain"].isna().all()
    assert df["ProductCD"].isna().tolist() == [False, True]


def test_missing_columns_become_nan():
    df = build_frame([rec()], FEATURES, CATEGORIES)
    assert df[["card1", "id_01", "P_emaildomain"]].isna().all().all()


@pytest.mark.parametrize("bad", ["isFraud", "TransactionDT", "TransactionID", "typo_col"])
def test_columns_the_model_does_not_use_are_rejected(bad):
    with pytest.raises(ValueError, match=bad):
        build_frame([rec(**{bad: 1})], FEATURES, CATEGORIES)


def test_non_numeric_value_in_numeric_column_is_rejected():
    with pytest.raises(ValueError):
        build_frame([rec(card1="abc")], FEATURES, CATEGORIES)


def test_numeric_values_are_cast_like_training_float32():
    df = build_frame([rec(card1=0.1)], FEATURES, CATEGORIES)
    assert df["card1"].iloc[0] == np.float32(0.1)


def test_explicit_none_is_missing_not_an_error():
    df = build_frame([rec(card1=None, id_01=None)], FEATURES, CATEGORIES)
    assert df[["card1", "id_01"]].isna().all().all()


def test_categories_for_a_column_the_model_lacks_are_rejected():
    with pytest.raises(ValueError, match="categor"):
        build_frame([rec()], FEATURES, {**CATEGORIES, "ghost": ["x"]})


@pytest.mark.parametrize("amount", [0, -5, float("inf")])
def test_request_rejects_bad_amounts(amount):
    with pytest.raises(ValidationError):
        PredictRequest(transactions=[{"TransactionAmt": amount, "ProductCD": "W"}])


def test_request_rejects_unknown_product_code():
    with pytest.raises(ValidationError):
        PredictRequest(transactions=[{"TransactionAmt": 5, "ProductCD": "X"}])


def test_request_batch_limits():
    ok = {"TransactionAmt": 5, "ProductCD": "W"}
    with pytest.raises(ValidationError):
        PredictRequest(transactions=[])
    with pytest.raises(ValidationError):
        PredictRequest(transactions=[ok] * (MAX_BATCH + 1))
    assert len(PredictRequest(transactions=[ok] * MAX_BATCH).transactions) == MAX_BATCH


def test_extra_columns_survive_into_records():
    req = PredictRequest(transactions=[{"TransactionAmt": 5, "ProductCD": "W", "card1": 99}])
    assert records_from_request(req) == [{"TransactionAmt": 5.0, "ProductCD": "W", "card1": 99}]
