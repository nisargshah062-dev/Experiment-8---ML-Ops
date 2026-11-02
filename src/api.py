"""FastAPI service that serves the Iris model and exposes metrics."""
import json
import time

import joblib
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest,
)
from pydantic import BaseModel, Field

from src.train import METRICS_PATH, MODEL_PATH, SPECIES

REQUESTS = Counter(
    "iris_http_requests_total", "HTTP requests", ["path", "status"],
)
LATENCY = Histogram(
    "iris_request_latency_seconds", "Request latency in seconds", ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)
PREDICTIONS = Counter(
    "iris_predictions_total", "Predictions made", ["species"],
)
CONFIDENCE = Histogram(
    "iris_prediction_confidence", "Probability of the predicted class",
    ["species"], buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
ACCURACY = Gauge("iris_model_accuracy", "Accuracy on the held-out test set")

model = joblib.load(MODEL_PATH)
model_metrics = json.loads(METRICS_PATH.read_text())
ACCURACY.set(model_metrics["accuracy"])

app = FastAPI(title="Iris Classifier API", version="1.0")


class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0, le=15, examples=[5.1])
    sepal_width: float = Field(gt=0, le=15, examples=[3.5])
    petal_length: float = Field(gt=0, le=15, examples=[1.4])
    petal_width: float = Field(gt=0, le=15, examples=[0.2])


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    if path != "/metrics":
        LATENCY.labels(path).observe(time.perf_counter() - start)
        REQUESTS.labels(path, str(response.status_code)).inc()
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {"model": "RandomForestClassifier", **model_metrics}


@app.post("/predict")
def predict(features: IrisFeatures):
    row = [[
        features.sepal_length, features.sepal_width,
        features.petal_length, features.petal_width,
    ]]
    probabilities = model.predict_proba(row)[0]
    best = int(probabilities.argmax())
    species = SPECIES[best]
    confidence = float(probabilities[best])
    PREDICTIONS.labels(species).inc()
    CONFIDENCE.labels(species).observe(confidence)
    return {
        "species": species,
        "confidence": round(confidence, 4),
        "probabilities": {
            name: round(float(p), 4) for name, p in zip(SPECIES, probabilities)
        },
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
