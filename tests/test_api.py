import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

JSON_PATH = Path(__file__).parent / "first_row.json"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    with open(JSON_PATH, "r") as f:
        expected = json.load(f)

    response = client.post(
        "/predict",
        json=expected["features"]
    )

    assert response.status_code == 200

    result = response.json()

    assert abs(
        result["probability"] - expected["predict_proba"]
    ) < 1e-6

    assert result["prediction"] == expected["prediction"]

    assert abs(
        result["threshold"] - expected["threshold"]
    ) < 1e-12