import json
from pathlib import Path

from app.model import predict_client


JSON_PATH = Path(__file__).parent / "first_row.json"


def test_predict_client():
    with open(JSON_PATH, "r") as f:
        expected = json.load(f)

    result = predict_client(expected["features"])

    assert abs(
        result["probability"] - expected["predict_proba"]
    ) < 1e-6

    assert result["prediction"] == expected["prediction"]

    assert abs(
        result["threshold"] - expected["threshold"]
    ) < 1e-6