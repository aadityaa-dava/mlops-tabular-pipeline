import pandas as pd

from mlops_pipeline.gbm import align_categoricals, split_xy


def test_rare_and_unseen_categories_become_nan():
    train = pd.DataFrame({"c": ["a"] * 5 + ["b"], "n": range(6)})
    val = pd.DataFrame({"c": ["a", "b", "z"], "n": [1, 2, 3]})
    t, v = align_categoricals(train, val, min_count=2)
    assert list(t["c"].cat.categories) == ["a"]
    assert v["c"].isna().tolist() == [False, True, True]
    assert v["c"].dtype == t["c"].dtype  # identical categories in every split


def test_split_xy_drops_id_time_and_label():
    df = pd.DataFrame(
        {"TransactionID": [1], "TransactionDT": [10], "isFraud": [1], "TransactionAmt": [5.0]}
    )
    x, y = split_xy(df)
    assert list(x.columns) == ["TransactionAmt"]
    assert y.tolist() == [1]


def test_build_lgbm_uses_the_given_seed():
    from mlops_pipeline.gbm import build_lgbm

    assert build_lgbm().get_params()["random_state"] == 42
    assert build_lgbm(7).get_params()["random_state"] == 7
