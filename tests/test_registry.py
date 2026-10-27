import pytest
from mlflow import MlflowClient

from mlops_pipeline.promotion import should_promote
from mlops_pipeline.registry import find_test_scoring_run, get_champion, promotion_metrics


@pytest.fixture
def client(tmp_path):
    return MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")


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


def test_no_champion_when_the_model_is_not_registered(client):
    assert get_champion(client, "missing-model") is None


def test_no_champion_when_the_model_has_no_alias(client):
    client.create_registered_model("m")
    assert get_champion(client, "m") is None


def test_test_window_guard_is_registry_wide(client):
    exp_id = client.create_experiment("e")
    first = client.create_run(exp_id)
    second = client.create_run(exp_id)
    assert find_test_scoring_run(client, exp_id) is None
    client.set_tag(first.info.run_id, "test_scored", "yes")
    # the guard finds the first run even when a different run is being considered
    assert find_test_scoring_run(client, exp_id) == first.info.run_id
    assert second.info.run_id != first.info.run_id


def test_guard_ignores_runs_not_tagged_yes(client):
    exp_id = client.create_experiment("e")
    run = client.create_run(exp_id)
    client.set_tag(run.info.run_id, "test_scored", "no")
    assert find_test_scoring_run(client, exp_id) is None
