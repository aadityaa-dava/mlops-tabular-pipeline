import pytest

from mlops_pipeline.promotion import should_promote, window_mismatch


def metrics(pr_auc: float, recall: float, diff: float = 0.0) -> dict[str, float]:
    return {"pr_auc": pr_auc, "recall_at_5pct": recall, "roundtrip_max_abs_diff": diff}


CHAMPION = metrics(0.6280, 0.6614)


def test_batch1_seed42_result_is_rejected():
    decision = should_promote(metrics(0.6347, 0.6644), champion=CHAMPION)  # +0.0067
    assert not decision.promote
    assert "below the required" in decision.reason


def test_gap14_result_is_rejected():
    assert not should_promote(metrics(0.5618, 0.6151), champion=CHAMPION).promote


def test_clear_gain_is_promoted():
    assert should_promote(metrics(0.6500, 0.6700), champion=CHAMPION).promote


def test_margin_boundary_is_inclusive_despite_float_error():
    # 0.11 - 0.1 is 0.00999...95 in floating point, but is exactly the margin
    assert should_promote(metrics(0.11, 0.5), champion=metrics(0.1, 0.5)).promote


def test_recall_guardrail_blocks_a_pr_auc_gain():
    decision = should_promote(metrics(0.65, 0.64), champion=metrics(0.60, 0.66))
    assert not decision.promote
    assert "recall" in decision.reason


def test_small_recall_drop_within_tolerance_is_allowed():
    assert should_promote(metrics(0.65, 0.655), champion=metrics(0.60, 0.66)).promote


def test_failed_roundtrip_blocks_promotion():
    decision = should_promote(metrics(0.70, 0.70, diff=1e-3), champion=CHAMPION)
    assert not decision.promote
    assert "reproduce" in decision.reason


def test_first_model_is_compared_with_the_baseline():
    baseline = metrics(0.3832, 0.4227)
    decision = should_promote(metrics(0.6280, 0.6614), baseline=baseline)
    assert decision.promote
    assert "baseline" in decision.reason


def test_needs_something_to_compare_against():
    with pytest.raises(ValueError):
        should_promote(metrics(0.6, 0.6))


WINDOW = {"train_end_day": "120.0", "val_end_day": "150.0", "val_rows": "85303"}


def test_same_window_is_comparable():
    assert window_mismatch(WINDOW, dict(WINDOW)) is None


def test_different_window_is_refused():
    other = {**WINDOW, "val_rows": "90000"}
    assert "val_rows" in window_mismatch(WINDOW, other)


def test_missing_window_info_is_refused():
    assert window_mismatch(WINDOW, {"train_end_day": "120.0"}) is not None
