import time

from fastapi import FastAPI
from pydantic import BaseModel

from app.model import predict_client
from app.monitoring import init_db, log_prediction


app = FastAPI(
    title="Credit Scoring API",
    version="1.0.0"
)

init_db()


class ClientData(BaseModel):
    EXT_SOURCE_3: float
    CLOSED_AMT_CREDIT_SUM_DEBT_MEAN: float
    CLOSED_AMT_CREDIT_SUM_DEBT_SUM: float
    CLOSED_AMT_CREDIT_SUM_DEBT_MAX: float
    BURO_DAYS_CREDIT_MEAN: float
    BURO_DAYS_CREDIT_ENDDATE_MIN: float
    CC_MONTHS_BALANCE_VAR: float
    BURO_DAYS_CREDIT_MIN: float
    CLOSED_DAYS_CREDIT_ENDDATE_MIN: float
    ACTIVE_DAYS_CREDIT_MIN: float
    CC_AMT_PAYMENT_CURRENT_MIN: float
    CC_AMT_PAYMENT_CURRENT_MEAN: float
    CC_AMT_DRAWINGS_POS_CURRENT_MIN: float
    REFUSED_APP_CREDIT_PERC_MEAN: float
    REFUSED_APP_CREDIT_PERC_MIN: float
    REFUSED_APP_CREDIT_PERC_MAX: float
    APPROVED_APP_CREDIT_PERC_VAR: float
    CC_AMT_PAYMENT_TOTAL_CURRENT_MIN: float
    CC_AMT_PAYMENT_CURRENT_MAX: float
    CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN: float


@app.get("/")
def root():
    return {"message": "Credit Scoring API"}


@app.get("/health")
def health():
    return {"status": "ok"}


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