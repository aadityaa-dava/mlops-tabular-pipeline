import numpy as np
import pandas as pd

from mlops_pipeline.baseline import build_model, make_features


def make_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "TransactionAmt": rng.uniform(1, 500, n),
            "TransactionDT": np.arange(n) * 3600 * 5,
            "id_01": np.where(np.arange(n) % 3 == 0, -5.0, np.nan),
            "ProductCD": rng.choice(["W", "C", "R"], n),
            "isFraud": [0, 1] * (n // 2),
        }
    )
    for i in range(1, 15):
        df[f"C{i}"] = rng.integers(0, 10, n).astype(float)
    df.loc[0, "C1"] = -3.0  # negative counts must be clipped, not produce NaN
    return df


def test_features_exclude_raw_time_and_have_no_nans_in_c_cols():
    feats = make_features(make_df())
    assert "TransactionDT" not in feats.columns
    assert "TransactionID" not in feats.columns
    assert not feats["C1"].isna().any()


def test_has_identity_flag():
    feats = make_features(make_df())
    assert feats["has_identity"].sum() == 20  # every third of 60 rows


def test_pipeline_fits_and_predicts_probabilities():
    df = make_df()
    model = build_model().fit(make_features(df), df["isFraud"])
    p = model.predict_proba(make_features(df))[:, 1]
    assert p.shape == (60,)
    assert ((p >= 0) & (p <= 1)).all()
