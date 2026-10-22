import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model import (
    predict_client,
    explain_client,
)
from app.monitoring import (
    init_db,
    log_prediction,
)


app = FastAPI(
    title="Credit Scoring API",
    version="1.0.0",
)

init_db()


class ClientData(BaseModel):

    SK_ID_CURR: int = Field(gt=0)

    EXT_SOURCE_3: float = Field(
        ge=0,
        le=1
    )

    APPROVED_APP_CREDIT_PERC_VAR: float = Field(ge=0)

    CC_AMT_DRAWINGS_POS_CURRENT_SUM: float = Field(ge=0)
    CC_AMT_DRAWINGS_POS_CURRENT_MAX: float = Field(ge=0)
    CC_AMT_DRAWINGS_POS_CURRENT_MEAN: float = Field(ge=0)
    CC_AMT_DRAWINGS_POS_CURRENT_MIN: float = Field(ge=0)

    CC_AMT_PAYMENT_CURRENT_SUM: float = Field(ge=0)
    CC_AMT_PAYMENT_CURRENT_MEAN: float = Field(ge=0)
    CC_AMT_PAYMENT_CURRENT_MAX: float = Field(ge=0)
    CC_AMT_PAYMENT_CURRENT_MIN: float = Field(ge=0)

    CC_AMT_PAYMENT_TOTAL_CURRENT_SUM: float = Field(ge=0)
    CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN: float = Field(ge=0)
    CC_AMT_PAYMENT_TOTAL_CURRENT_MAX: float = Field(ge=0)
    CC_AMT_PAYMENT_TOTAL_CURRENT_MIN: float = Field(ge=0)

    CC_AMT_DRAWINGS_CURRENT_SUM: float = Field(ge=0)
    CC_AMT_DRAWINGS_CURRENT_MEAN: float = Field(ge=0)
    CC_AMT_DRAWINGS_CURRENT_MAX: float = Field(ge=0)
    CC_AMT_DRAWINGS_CURRENT_MIN: float = Field(ge=0)

    CLOSED_DAYS_CREDIT_ENDDATE_MIN: float
    CLOSED_DAYS_CREDIT_MEAN: float = Field(le=0)


@app.get("/")
def root():
    return {
        "message": "Credit Scoring API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(data: ClientData):

    start = time.perf_counter()

    features = data.model_dump()

    result = predict_client(features)

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    log_prediction(
        features=features,
        probability=result["probability"],
        prediction=result["prediction"],
        latency_ms=latency_ms,
    )

    return {
        **result,
        "latency_ms": latency_ms,
    }


@app.post("/explain")
def explain(data: ClientData):
    """
    Retourne l'explication SHAP locale du client.

    Cet endpoint est séparé de /predict afin de ne pas
    ralentir toutes les prédictions avec un calcul SHAP.
    """

    return explain_client(
        data.model_dump()
    )