import pandas as pd

from mlops_pipeline.config import ModelConfig
from mlops_pipeline.gbm import align_categoricals, build_lgbm, split_xy, trim_train


def make_model_cfg() -> ModelConfig:
    return ModelConfig(
        seed=42,
        early_stopping_rounds=10,
        min_category_count=2,
        params={"n_estimators": 3, "num_leaves": 4, "learning_rate": 0.1},
    )


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


def test_build_lgbm_uses_config_params_and_seed():
    params = build_lgbm(make_model_cfg()).get_params()
    assert params["random_state"] == 42
    assert (params["n_estimators"], params["num_leaves"], params["learning_rate"]) == (3, 4, 0.1)


def test_build_lgbm_seed_override():
    assert build_lgbm(make_model_cfg(), seed=7).get_params()["random_state"] == 7


def test_trim_train_gap_and_early_trim():
    df = pd.DataFrame({"TransactionDT": [d * 86400 for d in [1, 10, 100, 110, 119]]})
    assert len(trim_train(df, 0.0, 0.0, 120.0)) == 5
    assert len(trim_train(df, 10.0, 0.0, 120.0)) == 3  # keeps days < 110
    assert len(trim_train(df, 0.0, 10.0, 120.0)) == 4  # keeps days >= 10
    assert len(trim_train(df, 10.0, 10.0, 120.0)) == 2
