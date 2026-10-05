from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import json


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "production.db"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                features TEXT NOT NULL,
                probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                latency_ms REAL NOT NULL
            )
            """
        )


def log_prediction(
    features: dict,
    probability: float,
    prediction: int,
    latency_ms: float
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                timestamp,
                features,
                probability,
                prediction,
                latency_ms
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(features),
                probability,
                prediction,
                latency_ms,
            )
        )