import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops_pipeline.api import LoadedModel, create_app
from mlops_pipeline.serving import MAX_BATCH

FEATURES = ["TransactionAmt", "card1", "ProductCD", "P_emaildomain"]
CATEGORIES = {"ProductCD": ["W", "C"], "P_emaildomain": ["gmail.com"]}


class FakeModel:
    """Scores amount / 1000, and remembers the last frame it was given."""

    def __init__(self):
        self.seen = None

    def predict_proba(self, x):
        self.seen = x
        p = np.clip(x["TransactionAmt"].to_numpy(dtype=float) / 1000, 0, 1)
        return np.column_stack([1 - p, p])


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def client(model):
    loaded = LoadedModel(model, FEATURES, CATEGORIES, version="7")
    with TestClient(create_app(lambda: loaded)) as c:
        yield c


def tx(amount=250.0, **extra):
    return {"TransactionAmt": amount, "ProductCD": "W", **extra}


def test_health_reports_model_version_and_feature_count(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_version": "7", "n_features": 4}


def test_predict_returns_one_score_per_transaction_in_order(client):
    body = {"transactions": [tx(250.0), tx(500.0), tx(100.0)]}
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
    assert resp.json()["scores"] == pytest.approx([0.25, 0.5, 0.1])
    assert resp.json()["model_version"] == "7"


def test_unseen_category_reaches_the_model_as_missing(client, model):
    body = {"transactions": [tx(P_emaildomain="weird.example", card1=5)]}
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
    assert model.seen["P_emaildomain"].isna().all()
    assert list(model.seen.columns) == FEATURES


def test_unknown_column_is_rejected_with_its_name(client):
    resp = client.post("/predict", json={"transactions": [tx(isFraud=1)]})
    assert resp.status_code == 422
    assert "isFraud" in resp.text


@pytest.mark.parametrize("amount", [0, -5])
def test_bad_amount_is_rejected(client, amount):
    resp = client.post("/predict", json={"transactions": [tx(amount)]})
    assert resp.status_code == 422


def test_missing_required_field_is_rejected(client):
    resp = client.post("/predict", json={"transactions": [{"TransactionAmt": 5.0}]})
    assert resp.status_code == 422


def test_empty_and_oversized_batches_are_rejected(client):
    assert client.post("/predict", json={"transactions": []}).status_code == 422
    too_many = {"transactions": [tx()] * (MAX_BATCH + 1)}
    assert client.post("/predict", json=too_many).status_code == 422


def test_non_numeric_value_is_rejected(client):
    resp = client.post("/predict", json={"transactions": [tx(card1="abc")]})
    assert resp.status_code == 422


def test_app_refuses_to_start_without_a_model():
    def no_model():
        raise RuntimeError("no champion")

    with pytest.raises(RuntimeError, match="no champion"):
        with TestClient(create_app(no_model)):
            pass
