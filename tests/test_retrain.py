import numpy as np
import pytest

from mlops_pipeline.retrain import decision_report, paired_bootstrap, recall_drop


def make_scores(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    base = rng.random(n) * 0.5 + y * 0.2
    return y, base


def test_a_better_scorer_has_a_positive_paired_interval():
    y, base = make_scores()
    better = base + y * 0.3
    diff, low, high = paired_bootstrap(y, better, base, n_boot=50)
    assert diff > 0 and low > 0 and high >= low


def test_identical_scores_give_a_zero_difference_and_a_zero_interval():
    y, base = make_scores()
    diff, low, high = paired_bootstrap(y, base, base, n_boot=50)
    assert diff == 0 and low == 0 and high == 0


def test_paired_bootstrap_is_reproducible():
    y, base = make_scores()
    other = base + np.random.default_rng(3).normal(0, 0.05, len(base))
    first = paired_bootstrap(y, other, base, n_boot=30)
    assert first == paired_bootstrap(y, other, base, n_boot=30)


def test_recall_drop_is_positive_when_the_candidate_is_worse():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    good = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    bad = np.array([0.1, 0.1, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert recall_drop(y, candidate=bad, champion=good, budget=0.2) == pytest.approx(1.0)
    assert recall_drop(y, candidate=good, champion=good, budget=0.2) == 0.0


def test_decision_report_carries_the_rule_verdict_and_the_metadata():
    rejected = decision_report(0.0036, -0.0027, 0.0111, 0.0, 0.0, {"champion_version": "1"})
    assert rejected["promote"] is False
    assert rejected["champion_version"] == "1"
    accepted = decision_report(0.0878, 0.0776, 0.0981, 0.0, 0.0, {})
    assert accepted["promote"] is True
    assert accepted["interval"] == [0.0776, 0.0981]
