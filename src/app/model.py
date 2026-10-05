from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "model.pkl"

model = joblib.load(MODEL_PATH)


MODEL_FEATURES = [
    "EXT_SOURCE_3",
    "CLOSED_AMT_CREDIT_SUM_DEBT_MEAN",
    "CLOSED_AMT_CREDIT_SUM_DEBT_SUM",
    "CLOSED_AMT_CREDIT_SUM_DEBT_MAX",
    "BURO_DAYS_CREDIT_MEAN",
    "BURO_DAYS_CREDIT_ENDDATE_MIN",
    "CC_MONTHS_BALANCE_VAR",
    "BURO_DAYS_CREDIT_MIN",
    "CLOSED_DAYS_CREDIT_ENDDATE_MIN",
    "ACTIVE_DAYS_CREDIT_MIN",
    "CC_AMT_PAYMENT_CURRENT_MIN",
    "CC_AMT_PAYMENT_CURRENT_MEAN",
    "CC_AMT_DRAWINGS_POS_CURRENT_MIN",
    "REFUSED_APP_CREDIT_PERC_MEAN",
    "REFUSED_APP_CREDIT_PERC_MIN",
    "REFUSED_APP_CREDIT_PERC_MAX",
    "APPROVED_APP_CREDIT_PERC_VAR",
    "CC_AMT_PAYMENT_TOTAL_CURRENT_MIN",
    "CC_AMT_PAYMENT_CURRENT_MAX",
    "CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN",
]


def predict_client(features: dict) -> dict:

    X = pd.DataFrame(
        [[features[col] for col in MODEL_FEATURES]],
        columns=MODEL_FEATURES
    )

    probability = model.predict_proba(X)[0, 1]

    prediction = int(
        probability >= 0.5
    )

    return {
        "probability": float(probability),
        "prediction": prediction,
        "threshold": 0.5
    }