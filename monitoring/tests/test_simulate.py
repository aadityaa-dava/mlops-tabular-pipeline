import numpy as np
import pandas as pd

from simulate import blank_fraction, first_detected, replace_fraction, scale_column, shift_mean


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": list("abcdefghij"),
        }
    )


def test_scale_changes_only_the_target_and_does_not_mutate_the_input():
    original = frame()
    out = scale_column(original, "a", 2.0)
    assert out["a"].dropna().tolist() == [2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    assert out["b"].tolist() == original["b"].tolist()
    assert original["a"].iloc[0] == 1.0


def test_shift_mean_moves_present_values_and_keeps_missing_missing():
    out = shift_mean(frame(), "a", 0.5, 4.0)
    assert out["a"].iloc[0] == 3.0  # 1 + 0.5 * 4
    assert np.isnan(out["a"].iloc[2])


def test_blank_fraction_blanks_an_exact_share_of_present_values():
    out = blank_fraction(frame(), "a", 0.3, seed=1)  # 9 present values, round(2.7) = 3
    assert out["a"].isna().sum() == 1 + 3
    assert np.isnan(out["a"].iloc[2])  # the value that was already missing stays missing


def test_blank_fraction_is_deterministic_and_zero_changes_nothing():
    first = blank_fraction(frame(), "a", 0.3, seed=1)
    second = blank_fraction(frame(), "a", 0.3, seed=1)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(blank_fraction(frame(), "a", 0.0), frame())


def test_replace_fraction_sets_an_exact_number_of_rows():
    out = replace_fraction(frame(), "b", 0.3, "new", seed=2)
    assert (out["b"] == "new").sum() == 3


def test_first_detected_returns_the_smallest_detected_magnitude():
    rows = [
        {"magnitude": 1.5, "target_detected": True},
        {"magnitude": 1.1, "target_detected": False},
        {"magnitude": 1.2, "target_detected": True},
    ]
    assert first_detected(rows) == 1.2


def test_first_detected_is_none_when_nothing_trips():
    assert first_detected([{"magnitude": 1.1, "target_detected": False}]) is None
