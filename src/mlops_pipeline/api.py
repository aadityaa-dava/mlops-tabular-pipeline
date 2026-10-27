"""FastAPI service: loads the champion model once at startup and scores transactions."""

import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.lightgbm
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient
from pydantic import BaseModel

from mlops_pipeline.config import Config, load_config
from mlops_pipeline.registry import CHAMPION_ALIAS
from mlops_pipeline.serving import PredictRequest, build_frame, records_from_request
from mlops_pipeline.tracking import predict_scores


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    feature_names: list[str]
    categories: dict[str, list[str]]
    version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    n_features: int


class PredictResponse(BaseModel):
    model_version: str
    scores: list[float]


def load_champion(cfg: Config) -> LoadedModel:
    """Load the model holding the champion alias, plus its saved category lists."""
    name = cfg.mlflow.registered_model
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mv = MlflowClient().get_model_version_by_alias(name, CHAMPION_ALIAS)
    model = mlflow.lightgbm.load_model(f"models:/{name}/{mv.version}")
    cats_path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="categories.json"
    )
    return LoadedModel(
        model=model,
        feature_names=list(model.feature_name_),
        categories=json.loads(Path(cats_path).read_text()),
        version=str(mv.version),
    )


def create_app(loader: Callable[[], LoadedModel] | None = None) -> FastAPI:
    load = loader or (lambda: load_champion(load_config()))

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
