from mlops_pipeline.promotion import should_promote
from mlops_pipeline.registry import promotion_metrics


def test_promotion_metrics_strips_val_prefix_and_keeps_roundtrip():
    out = promotion_metrics(
        {"val_pr_auc": 0.6, "val_recall_at_5pct": 0.5, "roundtrip_max_abs_diff": 0.0}
    )
    assert out == {"pr_auc": 0.6, "recall_at_5pct": 0.5, "roundtrip_max_abs_diff": 0.0}


def test_test_metrics_are_never_used_for_the_decision():
    out = promotion_metrics({"val_pr_auc": 0.6, "test_pr_auc": 0.9, "best_iteration": 5})
    assert out["pr_auc"] == 0.6
    assert "test_pr_auc" not in out and "best_iteration" not in out


def test_missing_roundtrip_check_blocks_promotion():
    candidate = promotion_metrics({"val_pr_auc": 0.9, "val_recall_at_5pct": 0.9})
    reference = {"pr_auc": 0.3, "recall_at_5pct": 0.4, "roundtrip_max_abs_diff": 0.0}
    assert not should_promote(candidate, baseline=reference).promote
