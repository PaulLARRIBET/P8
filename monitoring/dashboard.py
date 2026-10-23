from pathlib import Path
import json
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import matplotlib.pyplot as plt
import numpy as np
import requests
import shap
import joblib


ROOT = Path(__file__).resolve().parents[1]
API_URL = "http://127.0.0.1:8000"
DB_PATH = ROOT / "data" / "production.db"
REFERENCE_PATH = ROOT / "data" / "reference.csv"
MODEL_PATH = ROOT / "model" / "model.pkl"
DRIFT_TRAIN_PATH = ROOT / "data" / "X_train_before_kNN.csv"
DRIFT_TEST_PATH = ROOT / "data" / "X_test_before_kNN.csv"


st.set_page_config(
    page_title="Credit Scoring Monitoring",
    layout="wide"
)

st.title("Credit Scoring - Monitoring")


@st.cache_data
def load_reference():
    return pd.read_csv(REFERENCE_PATH)


@st.cache_data
def load_drift_data(path, file_version):
    return pd.read_csv(path)


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
# Data Drift avec Evidently
# =========================

st.header("Data Drift")
st.caption(
    "Comparaison de X_train_before_kNN.csv (référence) et "
    "X_test_before_kNN.csv (test), avant imputation kNN."
)

feature_columns = [
    col
    for col in reference.columns
    if col in production.columns
    and col != "SK_ID_CURR"
]

drift_train = load_drift_data(DRIFT_TRAIN_PATH, DRIFT_TRAIN_PATH.stat().st_mtime_ns)
drift_test = load_drift_data(DRIFT_TEST_PATH, DRIFT_TEST_PATH.stat().st_mtime_ns)
drift_columns = [
    col for col in drift_train.columns
    if col in drift_test.columns and col != "SK_ID_CURR"
]

if not drift_columns:
    st.warning(
        "Aucune feature commune entre les fichiers train et test avant kNN."
    )
    st.stop()

reference_drift = drift_train[
    drift_columns
].copy()

production_drift = drift_test[
    drift_columns
].copy()

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

report.run(
    current_data=production_drift,
    reference_data=reference_drift
)

# Récupération des résultats directement en Python
report_dict = report.as_dict()


# =========================
# Extraction des résultats
# =========================

drift_rows = []

for metric in report_dict.get("metrics", []):

    result = metric.get("result", {})

    # Résultats par colonne
    drift_by_columns = result.get(
        "drift_by_columns",
        {}
    )

    for feature, values in drift_by_columns.items():

        drift_rows.append({
            "feature": feature,
            "drift_detected": values.get(
                "drift_detected"
            ),
            "drift_score": values.get(
                "drift_score"
            ),
            "threshold": values.get(
                "stattest_threshold"
            ),
            "test": values.get(
                "stattest_name"
            ),
        })


drift_df = pd.DataFrame(
    drift_rows
)


# =========================
# Affichage Streamlit
# =========================

if drift_df.empty:

    st.warning(
        "Impossible d'extraire les résultats "
        "de drift depuis Evidently."
    )

else:

    drift_df = drift_df.sort_values(
        by="drift_score",
        ascending=False
    )

    drift_rate = (
        drift_df["drift_detected"]
        .fillna(False)
        .mean()
        * 100
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Features analysées",
        len(drift_df)
    )

    col2.metric(
        "Features avec drift",
        f"{drift_rate:.1f}%"
    )

    st.subheader(
        "Drift par variable"
    )

    st.dataframe(
        drift_df,
        use_container_width=True
    )

    fig_drift = px.bar(
        drift_df,
        x="feature",
        y="drift_score",
        color="drift_detected",
        labels={
            "feature": "Feature",
            "drift_score": "Score de drift",
            "drift_detected": "Drift détecté",
        },
    )

    fig_drift.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_drift,
        use_container_width=True
    )

# =========================
# Inspection par feature
# =========================

st.subheader(
    "Comparer une feature"
)

selected_feature = st.selectbox(
    "Feature",
    drift_columns
)

comparison = pd.DataFrame({
    "Entraînement avant kNN": reference_drift[
        selected_feature
    ].dropna(),
    "Test avant kNN": production_drift[
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

# =========================
# Explicabilité globale SHAP
# =========================

@st.cache_data
def compute_global_explanation(sample, model_version):
    # La version du fichier invalide le cache lorsque le modèle change.
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["model"]
    features = list(pipeline.feature_names_in_)
    X = sample[features].copy()
    explainer = shap.TreeExplainer(pipeline.named_steps["lgbm"])
    values = explainer.shap_values(X)

    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]

    return X, values


st.header("Explicabilité globale")

st.write(
    "Identifiez les variables qui influencent le plus les scores du modèle "
    "sur les données d'entraînement de référence."
)

try:
    with st.spinner("Calcul de l'explicabilité globale…"):
        global_X, global_shap_values = compute_global_explanation(
            reference, MODEL_PATH.stat().st_mtime_ns
        )

    st.caption(
        f"Analyse des {len(global_X):,} lignes de données d'entraînement "
        "disponibles dans reference.csv, avec le modèle actuel. "
        "Les contributions SHAP sont exprimées en log-odds, "
        "pas en points de probabilité."
    )

    global_importance = pd.DataFrame({
        "Variable": global_X.columns,
        "Importance SHAP moyenne": np.abs(global_shap_values).mean(axis=0),
    }).sort_values("Importance SHAP moyenne", ascending=False)

    st.subheader("Importance globale des variables")
    fig_global = px.bar(
        global_importance.head(20).sort_values("Importance SHAP moyenne"),
        x="Importance SHAP moyenne",
        y="Variable",
        orientation="h",
    )
    st.plotly_chart(fig_global, use_container_width=True)

    st.subheader("Sens et dispersion des contributions")
    st.write(
        "Chaque point représente une prédiction. Une contribution positive "
        "augmente le score de classe 1 ; une contribution négative le diminue. "
        "Le rouge indique une valeur élevée de la variable, le bleu une valeur faible."
    )
    plt.figure()
    shap.summary_plot(
        global_shap_values, global_X, max_display=20, show=False
    )
    fig_summary = plt.gcf()
    st.pyplot(fig_summary, clear_figure=True)
    plt.close(fig_summary)

except (OSError, KeyError, ValueError) as error:
    st.warning(f"Impossible de calculer l'explicabilité globale : {error}")


# =========================
# Explicabilité locale SHAP
# =========================

st.header("Explicabilité locale")

st.write(
    "Sélectionnez un client pour comprendre "
    "les variables ayant influencé sa prédiction."
)

if "SK_ID_CURR" not in production.columns:

    st.warning(
        "SK_ID_CURR n'est pas disponible "
        "dans les données de production."
    )

else:

    client_ids = sorted(
        production["SK_ID_CURR"].astype(int).unique().tolist()
    )

    selected_client_id = st.selectbox(
        "ID_Client",
        client_ids,
        format_func=str,
        help="Sélectionnez un identifiant ou tapez son numéro pour le rechercher.",
    )

    # En cas de plusieurs prédictions, afficher la plus récente du client.
    selected_row = production.loc[
        production["SK_ID_CURR"] == selected_client_id
    ].sort_values(["timestamp", "id"]).iloc[-1]

    st.write(
        f"**Client :** "
        f"{int(selected_row['SK_ID_CURR'])}"
    )

    st.write(
        f"**Probabilité prédite :** "
        f"{selected_row['probability']:.4f}"
    )

    st.write(
        f"**Classe prédite :** "
        f"{int(selected_row['prediction'])}"
    )

    # Payload attendu par l'API
    payload = {
        "SK_ID_CURR": int(
            selected_row["SK_ID_CURR"]
        )
    }

    for feature in feature_columns:
        payload[feature] = float(
            selected_row[feature]
        )

    if st.button(
        "Afficher l'explication SHAP"
    ):

        try:

            response = requests.post(
                f"{API_URL}/explain",
                json=payload,
                timeout=30,
            )

            response.raise_for_status()

            explanation_data = response.json()

            # =========================
            # Top feature importance
            # =========================

            importance_df = pd.DataFrame(
                explanation_data[
                    "local_feature_importance"
                ]
            )

            st.subheader(
                "Variables les plus importantes"
            )

            st.dataframe(
                importance_df[
                    [
                        "feature",
                        "value",
                        "shap_value",
                        "importance",
                    ]
                ].head(10),
                use_container_width=True,
            )

            # =========================
            # SHAP Waterfall
            # =========================

            st.subheader(
                "SHAP Waterfall"
            )

            shap_explanation = shap.Explanation(
                values=np.array(
                    explanation_data[
                        "shap_values"
                    ]
                ),
                base_values=explanation_data[
                    "base_value"
                ],
                data=np.array(
                    explanation_data[
                        "feature_values"
                    ]
                ),
                feature_names=explanation_data[
                    "feature_names"
                ],
            )

            plt.figure()

            shap.plots.waterfall(
                shap_explanation,
                max_display=10,
                show=False,
            )

            fig = plt.gcf()

            st.pyplot(
                fig,
                clear_figure=True,
            )

            plt.close(fig)

        except requests.RequestException as error:

            st.error(
                "Impossible de récupérer "
                f"l'explication SHAP : {error}"
            )
