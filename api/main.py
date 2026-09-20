"""FastAPI inference service for the frequency-severity risk model.

Run with `uvicorn api.main:app --reload` from the project root.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.inference import ModelBundle, load_models, predict_single
from api.schemas import PolicyFeatures, PredictionResponse

_state: dict[str, ModelBundle] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once at startup, not per-request -- XGBoost model + SHAP explainer
    # construction is not free, and re-doing it on every call would dominate
    # request latency for no benefit.
    _state["bundle"] = load_models()
    yield
    _state.clear()


app = FastAPI(title="Claim Risk API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": "bundle" in _state}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PolicyFeatures) -> PredictionResponse:
    return predict_single(payload, _state["bundle"])
