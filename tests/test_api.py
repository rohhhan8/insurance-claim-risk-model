"""API tests use a real ModelBundle since inference logic isn't meaningfully
testable against stubs -- the point is to catch train/serve skew, which only shows
up when the real trained artifacts and encoders are exercised together."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID_PAYLOAD = {
    "Exposure": 0.8,
    "VehPower": 7,
    "VehAge": 5,
    "DrivAge": 35,
    "BonusMalus": 60,
    "VehBrand": "B12",
    "VehGas": "Diesel",
    "Density": 1200,
    "Region": "R82",
    "Area": "D",
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["models_loaded"] is True


def test_predict_valid_payload(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_of_claim"] <= 1.0
    assert body["expected_severity"] > 0
    assert body["expected_loss"] > 0
    assert len(body["top_shap_frequency"]) == 5
    assert len(body["top_shap_severity"]) == 5


def test_predict_rejects_underage_driver(client):
    payload = {**VALID_PAYLOAD, "DrivAge": 10}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_handles_unseen_category(client):
    payload = {**VALID_PAYLOAD, "VehBrand": "UNKNOWN_BRAND"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
