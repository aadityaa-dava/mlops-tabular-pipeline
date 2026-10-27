"""FastAPI service: loads an exported model folder once at startup and scores transactions."""

import os
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from mlops_pipeline.inference import predict_scores
from mlops_pipeline.model_store import DEFAULT_MODEL_DIR, LoadedModel, load_exported
from mlops_pipeline.serving import PredictRequest, build_frame, records_from_request


class HealthResponse(BaseModel):
    status: str
    model_version: str
    n_features: int


class PredictResponse(BaseModel):
    model_version: str
    scores: list[float]


def create_app(loader: Callable[[], LoadedModel] | None = None) -> FastAPI:
    """Build the app; by default it loads the folder named by MODEL_DIR."""
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
        loaded: LoadedModel = request.app.state.loaded
        records = records_from_request(body)
        try:
            frame = build_frame(records, loaded.feature_names, loaded.categories)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        scores = predict_scores(loaded.model, frame)
        return PredictResponse(
            model_version=loaded.version,
            scores=[float(s) for s in scores],
        )

    return app
