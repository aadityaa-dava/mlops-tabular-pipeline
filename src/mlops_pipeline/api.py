"""FastAPI service: loads an exported model folder once at startup and scores transactions."""

import json
import logging
import os
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from mlops_pipeline.inference import predict_scores
from mlops_pipeline.model_store import DEFAULT_MODEL_DIR, LoadedModel, load_exported
from mlops_pipeline.serving import PredictRequest, build_frame, records_from_request

logger = logging.getLogger("fraud_api")
STDOUT_HANDLER_NAME = "fraud_api_stdout"


class HealthResponse(BaseModel):
    status: str
    model_version: str
    n_features: int


class PredictResponse(BaseModel):
    model_version: str
    scores: list[float]


def configure_logging() -> None:
    """Send the service's own log lines to stdout, one JSON object per line."""
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(h.get_name() == STDOUT_HANDLER_NAME for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name(STDOUT_HANDLER_NAME)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def prediction_log_line(
    model_version: str,
    records: Sequence[dict],
    scores: Sequence[float],
    latency_ms: float,
    now: datetime | None = None,
) -> str:
    """One JSON line describing a /predict call. Aggregates only: no field values."""
    present = [sum(value is not None for value in r.values()) for r in records]
    return json.dumps(
        {
            "event": "predict",
            "ts": (now or datetime.now(UTC)).isoformat(),
            "model_version": model_version,
            "n_rows": len(scores),
            "fields_per_row_mean": round(sum(present) / len(present), 2),
            "score_min": round(min(scores), 6),
            "score_mean": round(sum(scores) / len(scores), 6),
            "score_max": round(max(scores), 6),
            "latency_ms": round(latency_ms, 1),
        }
    )


def create_app(loader: Callable[[], LoadedModel] | None = None) -> FastAPI:
    """Build the app; by default it loads the folder named by MODEL_DIR."""
    configure_logging()
    load = loader or (lambda: load_exported(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.loaded = load()  # no model, no service: fail at startup
        yield

    app = FastAPI(title="Fraud scoring API", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded: LoadedModel = request.app.state.loaded
        return HealthResponse(
            status="ok",
            model_version=loaded.version,
            n_features=len(loaded.feature_names),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        started = time.perf_counter()
        loaded: LoadedModel = request.app.state.loaded
        records = records_from_request(body)
        try:
            frame = build_frame(records, loaded.feature_names, loaded.categories)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        scores = [float(s) for s in predict_scores(loaded.model, frame)]
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(prediction_log_line(loaded.version, records, scores, elapsed_ms))
        return PredictResponse(model_version=loaded.version, scores=scores)

    return app
