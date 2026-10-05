from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "model.pkl"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = float(artifact["threshold"])

MODEL_FEATURES = list(model.feature_name_)


def predict_client(features: dict) -> dict:

    X = pd.DataFrame([features])

    # Garantit exactement les colonnes et l'ordre attendus par LightGBM
    X = X[MODEL_FEATURES]

    probability = model.predict_proba(X)[0, 1]

    prediction = int(
        probability >= threshold
    )

    return {
        "probability": float(probability),
        "prediction": prediction,
        "threshold": threshold,
    }