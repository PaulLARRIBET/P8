from pathlib import Path

import joblib
import pandas as pd
import shap


MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "model.pkl"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = float(artifact["threshold"])

MODEL_FEATURES = list(model.feature_name_)

explainer = shap.TreeExplainer(model)


def predict_client(features: dict) -> dict:

    X = pd.DataFrame([features])
    X = X[MODEL_FEATURES]

    probability = model.predict_proba(X)[0, 1]

    prediction = int(
        probability >= threshold
    )

    shap_values = explainer.shap_values(X)

    # Selon la version de SHAP / LightGBM
    if isinstance(shap_values, list):
        local_values = shap_values[1][0]
    else:
        local_values = shap_values[0]

    local_importance = [
        {
            "feature": feature,
            "value": float(X.iloc[0][feature]),
            "shap_value": float(shap_value),
            "importance": abs(float(shap_value)),
        }
        for feature, shap_value in zip(
            MODEL_FEATURES,
            local_values
        )
    ]

    local_importance = sorted(
        local_importance,
        key=lambda x: x["importance"],
        reverse=True
    )

    return {
        "probability": float(probability),
        "prediction": prediction,
        "threshold": threshold,
        "local_feature_importance": local_importance[:10]
    }