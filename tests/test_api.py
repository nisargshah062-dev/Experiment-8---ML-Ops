from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)
SETOSA = {
    "sepal_length": 5.1, "sepal_width": 3.5,
    "petal_length": 1.4, "petal_width": 0.2,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_species():
    response = client.post("/predict", json=SETOSA)
    assert response.status_code == 200
    assert response.json()["species"] == "setosa"


def test_predict_rejects_invalid_input():
    response = client.post("/predict", json={**SETOSA, "petal_width": -1})
    assert response.status_code == 422


def test_metrics_exposes_prediction_counter():
    client.post("/predict", json=SETOSA)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert 'iris_predictions_total{species="setosa"}' in response.text
