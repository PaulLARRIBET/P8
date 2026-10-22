from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "model.pkl"

artifact = joblib.load(MODEL_PATH)

# Pipeline complet : SMOTE + LightGBM
model = artifact["model"]

# Seuil métier
threshold = float(artifact["threshold"])

# Features attendues par le pipeline
MODEL_FEATURES = list(model.feature_names_in_)

# Récupération du vrai LightGBM à l'intérieur du pipeline
lgbm_model = model.named_steps["lgbm"]

# Explainer SHAP uniquement sur LightGBM
explainer = shap.TreeExplainer(lgbm_model)


def _build_model_input(features: dict) -> pd.DataFrame:
    """
    Construit les données envoyées au modèle.

    SK_ID_CURR est conservé pour identifier le client,
    mais n'est jamais envoyé au modèle.
    """

    return pd.DataFrame(
        [
            {
                feature: features[feature]
                for feature in MODEL_FEATURES
            }
        ]
    )


def predict_client(features: dict) -> dict:
    """
    Effectue une prédiction sans calculer SHAP.
    """

    client_id = int(features["SK_ID_CURR"])

    X = _build_model_input(features)

    # Pipeline complet.
    # SMOTE est automatiquement ignoré lors du predict.
    probability = model.predict_proba(X)[0, 1]

    prediction = int(
        probability >= threshold
    )

    return {
        "SK_ID_CURR": client_id,
        "probability": float(probability),
        "prediction": prediction,
        "threshold": threshold,
    }


def explain_client(features: dict) -> dict:
    """
    Calcule l'explication SHAP locale du LightGBM
    pour un client donné.
    """

    client_id = int(features["SK_ID_CURR"])

    X = _build_model_input(features)

    # SHAP explique directement le LightGBM.
    # SMOTE n'a aucun rôle lors de l'inférence.
    shap_values = explainer.shap_values(X)

    # Compatibilité entre différentes versions de SHAP
    if isinstance(shap_values, list):
        local_values = np.asarray(
            shap_values[1]
        )[0]

    else:
        shap_values = np.asarray(
            shap_values
        )

        if shap_values.ndim == 3:
            local_values = shap_values[
                0, :, 1
            ]
        else:
            local_values = shap_values[0]

    # Valeur de base SHAP
    expected_value = explainer.expected_value

    if isinstance(
        expected_value,
        (list, tuple, np.ndarray)
    ):
        expected_value = np.asarray(
            expected_value
        ).flatten()

        if len(expected_value) > 1:
            expected_value = expected_value[1]
        else:
            expected_value = expected_value[0]

    # Feature importance locale
    local_importance = [
        {
            "feature": feature,
            "value": float(
                X.iloc[0][feature]
            ),
            "shap_value": float(
                shap_value
            ),
            "importance": abs(
                float(shap_value)
            ),
        }
        for feature, shap_value in zip(
            MODEL_FEATURES,
            local_values
        )
    ]

    # Trier de la plus importante à la moins importante
    local_importance = sorted(
        local_importance,
        key=lambda x: x["importance"],
        reverse=True,
    )

    return {
        "SK_ID_CURR": client_id,
        "base_value": float(expected_value),

        "feature_names": MODEL_FEATURES,

        "feature_values": [
            float(X.iloc[0][feature])
            for feature in MODEL_FEATURES
        ],

        "shap_values": [
            float(value)
            for value in local_values
        ],

        "local_feature_importance":
            local_importance,
    }