import json
import logging
from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops_pipeline.api import STDOUT_HANDLER_NAME, create_app, prediction_log_line
from mlops_pipeline.model_store import LoadedModel

FEATURES = ["TransactionAmt", "card1", "ProductCD"]
CATEGORIES = {"ProductCD": ["W", "C"]}


class FakeModel:
    def predict_proba(self, x):
        p = np.clip(x["TransactionAmt"].to_numpy(dtype=float) / 1000, 0, 1)
        return np.column_stack([1 - p, p])


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def captured():
    handler = ListHandler()
    logger = logging.getLogger("fraud_api")
    logger.addHandler(handler)
    yield handler.lines
    logger.removeHandler(handler)


@pytest.fixture
def client(captured):
    loaded = LoadedModel(FakeModel(), FEATURES, CATEGORIES, version="7")
    with TestClient(create_app(lambda: loaded)) as c:
        yield c


def test_log_line_holds_aggregates_and_no_field_values():
    records = [
        {"TransactionAmt": 250.0, "ProductCD": "W", "card1": 9500, "P_emaildomain": None},
        {"TransactionAmt": 500.0, "ProductCD": "C"},
    ]
    now = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
    line = prediction_log_line("7", records, [0.25, 0.5], 12.34, now=now)
    data = json.loads(line)
    assert data == {
        "event": "predict",
        "ts": "2026-10-28T12:00:00+00:00",
        "model_version": "7",
        "n_rows": 2,
        "fields_per_row_mean": 2.5,
        "score_min": 0.25,
        "score_mean": 0.375,
        "score_max": 0.5,
        "latency_ms": 12.3,
    }
    assert "9500" not in line and "card1" not in line


def test_predict_logs_exactly_one_line_without_raw_values(client, captured):
    body = {"transactions": [{"TransactionAmt": 250.0, "ProductCD": "W", "card1": 9500}]}
    assert client.post("/predict", json=body).status_code == 200
    assert len(captured) == 1
    data = json.loads(captured[0])
    assert data["event"] == "predict"
    assert data["model_version"] == "7"
    assert data["n_rows"] == 1
    assert data["latency_ms"] >= 0
    assert "9500" not in captured[0]


def test_health_and_rejected_requests_log_nothing(client, captured):
    assert client.get("/health").status_code == 200
    bad = {"transactions": [{"TransactionAmt": 5.0, "ProductCD": "W", "isFraud": 1}]}
    assert client.post("/predict", json=bad).status_code == 422
    assert captured == []


def test_creating_the_app_twice_adds_one_stdout_handler(captured):
    loaded = LoadedModel(FakeModel(), FEATURES, CATEGORIES, version="7")
    create_app(lambda: loaded)
    create_app(lambda: loaded)
    handlers = logging.getLogger("fraud_api").handlers
    assert sum(h.get_name() == STDOUT_HANDLER_NAME for h in handlers) == 1
    assert logging.getLogger("fraud_api").propagate is False
