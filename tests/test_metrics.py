import pytest

from mlops_pipeline.metrics import evaluate, top_budget_stats


def test_top_budget_hand_computed():
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.05, 0.04, 0.03, 0.02]
    # budget 0.2 flags the top 2 scores: (0.9 -> fraud, 0.8 -> legit)
    recall, precision = top_budget_stats(y, s, budget=0.2)
    assert recall == pytest.approx(0.5)
    assert precision == pytest.approx(0.5)


def test_perfect_scores_give_pr_auc_one():
    result = evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], budget=0.5)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_constant_scores_pr_auc_equals_base_rate():
    result = evaluate([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], budget=0.5)
    assert result["pr_auc"] == pytest.approx(0.25)


def test_evaluate_default_keys():
    result = evaluate([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert set(result) == {"pr_auc", "roc_auc", "recall_at_1%", "precision_at_1%"}
