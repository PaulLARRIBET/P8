from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "data" / "production.db"
REFERENCE_PATH = ROOT / "data" / "reference.csv"


st.set_page_config(
    page_title="Credit Scoring Monitoring",
    layout="wide"
)

st.title("Credit Scoring - Monitoring")


@st.cache_data
def load_reference():
    return pd.read_csv(REFERENCE_PATH)


@st.cache_data
def load_production():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY timestamp",
            conn
        )

    if df.empty:
        return df

    feature_df = pd.json_normalize(
        df["features"].apply(json.loads)
    )

    df = pd.concat(
        [
            df.drop(columns=["features"]),
            feature_df
        ],
        axis=1
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    return df


reference = load_reference()
production = load_production()


if production.empty:
    st.warning(
        "Aucune donnée de production enregistrée pour le moment."
    )
    st.stop()


# =========================
# KPIs
# =========================

total_predictions = len(production)

positive_rate = (
    production["prediction"].mean() * 100
)

mean_latency = production[
    "latency_ms"
].mean()

p95_latency = production[
    "latency_ms"
].quantile(0.95)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Prédictions",
    f"{total_predictions:,}"
)

col2.metric(
    "Classe 1",
    f"{positive_rate:.1f}%"
)

col3.metric(
    "Latence moyenne",
    f"{mean_latency:.2f} ms"
)

col4.metric(
    "Latence P95",
    f"{p95_latency:.2f} ms"
)


# =========================
# Distribution des scores
# =========================

st.subheader(
    "Distribution des probabilités prédites"
)

fig_scores = px.histogram(
    production,
    x="probability",
    nbins=30,
    labels={
        "probability": "Probabilité prédite"
    }
)

st.plotly_chart(
    fig_scores,
    use_container_width=True
)


# =========================
# Répartition des classes
# =========================

st.subheader(
    "Répartition des prédictions"
)

class_counts = (
    production["prediction"]
    .value_counts()
    .sort_index()
    .rename_axis("prediction")
    .reset_index(name="count")
)

fig_classes = px.bar(
    class_counts,
    x="prediction",
    y="count"
)

st.plotly_chart(
    fig_classes,
    use_container_width=True
)


# =========================
# Evolution des scores
# =========================

st.subheader(
    "Evolution des scores dans le temps"
)

fig_time = px.line(
    production,
    x="timestamp",
    y="probability"
)

st.plotly_chart(
    fig_time,
    use_container_width=True
)


# =========================
# Latence
# =========================

st.subheader(
    "Distribution de la latence"
)

fig_latency = px.histogram(
    production,
    x="latency_ms",
    nbins=30
)

st.plotly_chart(
    fig_latency,
    use_container_width=True
)


# =========================
# Data Drift
# =========================

st.header("Data Drift")

feature_columns = [
    col
    for col in reference.columns
    if col in production.columns
]

drift_results = []

for feature in feature_columns:

    ref = reference[feature].dropna()
    prod = production[feature].dropna()

    if len(ref) == 0 or len(prod) == 0:
        continue

    statistic, p_value = ks_2samp(
        ref,
        prod
    )

    drift_results.append({
        "feature": feature,
        "ks_statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < 0.05
    })


drift_df = pd.DataFrame(
    drift_results
)

if not drift_df.empty:

    drift_df = drift_df.sort_values(
        "ks_statistic",
        ascending=False
    )

    st.dataframe(
        drift_df,
        use_container_width=True
    )

    drift_rate = (
        drift_df["drift_detected"].mean()
        * 100
    )

    st.metric(
        "Features avec drift détecté",
        f"{drift_rate:.1f}%"
    )


# =========================
# Inspection par feature
# =========================

st.subheader(
    "Comparer une feature"
)

selected_feature = st.selectbox(
    "Feature",
    feature_columns
)

comparison = pd.DataFrame({
    "reference": reference[
        selected_feature
    ].dropna(),
    "production": production[
        selected_feature
    ].dropna()
})

fig_feature = px.histogram(
    comparison,
    barmode="overlay",
    nbins=30
)

st.plotly_chart(
    fig_feature,
    use_container_width=True
)