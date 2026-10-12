import time
import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from app.model import predict_client
from app.monitoring import init_db, log_prediction


app = FastAPI(
    title="Credit Scoring API",
    version="1.0.0"
)

init_db()

ROOT = Path(__file__).resolve().parents[2]
FIRST_ROW_PATH = ROOT / "tests" / "first_row.json"

with open(FIRST_ROW_PATH, "r") as f:
    FIRST_ROW = json.load(f)

DEFAULT_FEATURES = FIRST_ROW["features"]

class ClientData(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={
            "example": DEFAULT_FEATURES
        }
    )

    EXT_SOURCE_3: float = Field(ge=0, le=1)

    CLOSED_AMT_CREDIT_SUM_DEBT_MEAN: float = Field(ge=0)
    CLOSED_AMT_CREDIT_SUM_DEBT_SUM: float = Field(ge=0)
    CLOSED_AMT_CREDIT_SUM_DEBT_MAX: float = Field(ge=0)

    BURO_DAYS_CREDIT_MEAN: float = Field(le=0)
    BURO_DAYS_CREDIT_ENDDATE_MIN: float
    CC_MONTHS_BALANCE_VAR: float = Field(ge=0)
    BURO_DAYS_CREDIT_MIN: float = Field(le=0)
    CLOSED_DAYS_CREDIT_ENDDATE_MIN: float
    ACTIVE_DAYS_CREDIT_MIN: float = Field(le=0)

    CC_AMT_PAYMENT_CURRENT_MIN: float = Field(ge=0)
    CC_AMT_PAYMENT_CURRENT_MEAN: float = Field(ge=0)
    CC_AMT_DRAWINGS_POS_CURRENT_MIN: float = Field(ge=0)

    REFUSED_APP_CREDIT_PERC_MEAN: float = Field(ge=0)
    REFUSED_APP_CREDIT_PERC_MIN: float = Field(ge=0)
    REFUSED_APP_CREDIT_PERC_MAX: float = Field(ge=0)

    APPROVED_APP_CREDIT_PERC_VAR: float = Field(ge=0)

    CC_AMT_PAYMENT_TOTAL_CURRENT_MIN: float = Field(ge=0)
    CC_AMT_PAYMENT_CURRENT_MAX: float = Field(ge=0)
    CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN: float = Field(ge=0)

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