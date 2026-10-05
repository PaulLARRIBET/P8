# P8 — Credit Scoring API & MLOps

Ce projet met en production un modèle de scoring de crédit à travers une API FastAPI.

L'objectif est de proposer une chaîne MLOps simple comprenant :

- un modèle de Machine Learning entraîné avec LightGBM ;
- une API REST permettant d'obtenir un score de risque client ;
- un seuil de classification métier sauvegardé avec le modèle ;
- des tests unitaires automatisés ;
- une conteneurisation Docker ;
- un système de monitoring des prédictions ;
- une analyse du Data Drift ;
- un pipeline CI avec GitHub Actions.

---

## 1. Architecture du projet

```text
P8/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── reference.csv
│   ├── x_test.csv
│   └── production.db
│
├── model/
│   └── model.pkl
│
├── monitoring/
│   └── dashboard.py
│
├── scripts/
│   └── send_x_test_to_api.py
│
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── model.py
│       └── monitoring.py
│
├── tests/
│   ├── first_row.json
│   ├── test_api.py
│   └── test_model.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 2. Modèle de Machine Learning

Le modèle utilisé est un `LGBMClassifier`.

Il a été sélectionné après comparaison de plusieurs approches de classification, notamment :

- Régression Logistique ;
- Random Forest ;
- LightGBM ;
- LightGBM avec SMOTE.

Les hyperparamètres du modèle LightGBM ont ensuite été optimisés avec Optuna.

Le modèle prend en entrée 20 variables sélectionnées :

```text
EXT_SOURCE_3
CLOSED_AMT_CREDIT_SUM_DEBT_MEAN
CLOSED_AMT_CREDIT_SUM_DEBT_SUM
CLOSED_AMT_CREDIT_SUM_DEBT_MAX
BURO_DAYS_CREDIT_MEAN
BURO_DAYS_CREDIT_ENDDATE_MIN
CC_MONTHS_BALANCE_VAR
BURO_DAYS_CREDIT_MIN
CLOSED_DAYS_CREDIT_ENDDATE_MIN
ACTIVE_DAYS_CREDIT_MIN
CC_AMT_PAYMENT_CURRENT_MIN
CC_AMT_PAYMENT_CURRENT_MEAN
CC_AMT_DRAWINGS_POS_CURRENT_MIN
REFUSED_APP_CREDIT_PERC_MEAN
REFUSED_APP_CREDIT_PERC_MIN
REFUSED_APP_CREDIT_PERC_MAX
APPROVED_APP_CREDIT_PERC_VAR
CC_AMT_PAYMENT_TOTAL_CURRENT_MIN
CC_AMT_PAYMENT_CURRENT_MAX
CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN
```

Le modèle est stocké dans :

```text
model/model.pkl
```

Le fichier contient à la fois :

- le modèle LightGBM ;
- le seuil métier utilisé pour transformer la probabilité en classe.

Exemple :

```python
artifact = {
    "model": final_model,
    "threshold": best_threshold
}
```

Cela garantit que le modèle et son seuil sont toujours versionnés ensemble.

---

# 3. Seuil métier

Le seuil de classification n'est pas nécessairement égal à `0.5`.

Il a été optimisé en prenant en compte une fonction de coût métier dans laquelle :

```text
Coût = 10 × FN + FP
```

avec :

- `FN` : faux négatifs ;
- `FP` : faux positifs.

Un faux négatif est donc considéré comme 10 fois plus coûteux qu'un faux positif.

La décision finale est calculée ainsi :

```python
prediction = int(probability >= threshold)
```

---

# 4. API FastAPI

L'API permet d'envoyer les données d'un client et de récupérer :

- sa probabilité de risque ;
- sa classe prédite ;
- le seuil utilisé ;
- la latence de la requête.

L'application se trouve dans :

```text
src/app/
```

## Lancer l'API

Depuis la racine du projet :

```bash
uv run uvicorn app.main:app --app-dir src --reload
```

L'API est alors disponible sur :

```text
http://127.0.0.1:8000
```

---

## Documentation Swagger

FastAPI génère automatiquement une documentation interactive.

Elle est disponible sur :

```text
http://127.0.0.1:8000/docs
```

Elle permet notamment de tester directement l'endpoint `/predict`.

---

# 5. Endpoints

## Health check

```http
GET /health
```

Réponse :

```json
{
  "status": "ok"
}
```

Cet endpoint permet de vérifier rapidement que l'API fonctionne.

---

## Prediction

```http
POST /predict
```

Exemple de requête :

```json
{
  "EXT_SOURCE_3": 0.1595195404777181,
  "CLOSED_AMT_CREDIT_SUM_DEBT_MEAN": 51881.85,
  "CLOSED_AMT_CREDIT_SUM_DEBT_SUM": 62219.7,
  "CLOSED_AMT_CREDIT_SUM_DEBT_MAX": 62219.7,
  "BURO_DAYS_CREDIT_MEAN": -772.6666666666667,
  "BURO_DAYS_CREDIT_ENDDATE_MIN": -435.4,
  "CC_MONTHS_BALANCE_VAR": 732.6,
  "BURO_DAYS_CREDIT_MIN": -1444.0,
  "CLOSED_DAYS_CREDIT_ENDDATE_MIN": -876.6,
  "ACTIVE_DAYS_CREDIT_MIN": -731.6,
  "CC_AMT_PAYMENT_CURRENT_MIN": 7571.835,
  "CC_AMT_PAYMENT_CURRENT_MEAN": 15395.9175,
  "CC_AMT_DRAWINGS_POS_CURRENT_MIN": 694.602,
  "REFUSED_APP_CREDIT_PERC_MEAN": 0.7886358501628111,
  "REFUSED_APP_CREDIT_PERC_MIN": 0.7886358501628111,
  "REFUSED_APP_CREDIT_PERC_MAX": 0.7886358501628111,
  "APPROVED_APP_CREDIT_PERC_VAR": 0.00224587158067247,
  "CC_AMT_PAYMENT_TOTAL_CURRENT_MIN": 4151.835,
  "CC_AMT_PAYMENT_CURRENT_MAX": 23220.0,
  "CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN": 10998.657
}
```

Exemple de réponse :

```json
{
  "probability": 0.05686984283426618,
  "prediction": 0,
  "threshold": 0.082,
  "latency_ms": 4.2
}
```

---

# 6. Installation

Le projet utilise `uv` pour la gestion de l'environnement Python et des dépendances.

## Prérequis

- Python 3.12
- uv
- Docker Desktop pour la conteneurisation

Cloner le repository :

```bash
git clone https://github.com/PaulLARRIBET/P8.git
cd P8
```

Installer les dépendances :

```bash
uv sync
```

---

# 7. Tests unitaires

Les tests sont situés dans :

```text
tests/
```

Ils vérifient notamment :

- que l'API répond correctement ;
- que l'endpoint `/health` fonctionne ;
- que `/predict` retourne une probabilité valide ;
- que la classe obtenue est cohérente avec le seuil métier ;
- que les prédictions sont identiques à une prédiction de référence.

Une ligne réelle de test est stockée dans :

```text
tests/first_row.json
```

Ce fichier contient :

- les features ;
- la probabilité attendue ;
- la classe attendue ;
- le seuil attendu.

Cela permet de réaliser un test de non-régression du modèle.

## Lancer les tests

```bash
uv run pytest -v
```

---

# 8. Docker

Le projet peut être exécuté dans un conteneur Docker.

Le `Dockerfile` installe notamment :

- Python 3.12 ;
- les dépendances Python ;
- LightGBM ;
- `libgomp1`, nécessaire au fonctionnement de LightGBM sous Linux ;
- le code de l'API ;
- le modèle.

## Construire l'image

```bash
docker build -t p8-credit-api .
```

## Lancer le conteneur

```bash
docker run --rm -p 8000:8000 p8-credit-api
```

L'API devient disponible sur :

```text
http://127.0.0.1:8000
```

Documentation Swagger :

```text
http://127.0.0.1:8000/docs
```

---

# 9. Monitoring des prédictions

Chaque appel à `/predict` est enregistré dans une base SQLite :

```text
data/production.db
```

Les informations stockées incluent :

- timestamp ;
- données client ;
- probabilité prédite ;
- classe prédite ;
- latence de l'API.

Ces données permettent de suivre le comportement du modèle en production.

---

# 10. Dashboard de monitoring

Un dashboard Streamlit est disponible dans :

```text
monitoring/dashboard.py
```

Pour le lancer :

```bash
uv run streamlit run monitoring/dashboard.py
```

Il est ensuite accessible, généralement, sur :

```text
http://localhost:8501
```

Le dashboard permet de suivre :

- le nombre total de prédictions ;
- la répartition des classes prédites ;
- la distribution des probabilités ;
- la latence moyenne ;
- la latence P95 ;
- l'évolution des scores dans le temps ;
- le Data Drift entre les données de référence et les données de production.

---

# 11. Data Drift

Le Data Drift correspond à une modification de la distribution des données reçues par le modèle par rapport aux données utilisées lors de son entraînement.

Dans ce projet :

```text
X_train = données de référence
X_test = données assimilées aux données de production
```

Le dataset de référence est stocké dans :

```text
data/reference.csv
```

Les données issues de `X_test` sont envoyées à l'API pour simuler l'utilisation réelle du modèle.

Le drift est analysé variable par variable.

Le dashboard compare les distributions de référence et de production à l'aide notamment du test statistique de Kolmogorov-Smirnov.

Un drift détecté ne signifie pas automatiquement que le modèle est devenu mauvais. Il indique que les données rencontrées par le modèle ont changé et qu'une surveillance supplémentaire peut être nécessaire.

---

# 12. Simulation des données de production

Le script :

```text
scripts/send_x_test_to_api.py
```

permet d'envoyer automatiquement les observations de `X_test` à l'API.

Pour l'utiliser, commencer par lancer l'API :

```bash
uv run uvicorn app.main:app --app-dir src --reload
```

Puis, dans un autre terminal :

```bash
uv run python scripts/send_x_test_to_api.py
```

Chaque observation est alors :

1. envoyée à l'API ;
2. scorée par le modèle ;
3. enregistrée dans la base de monitoring.

---

# 13. CI — GitHub Actions

Le projet possède un pipeline d'intégration continue dans :

```text
.github/workflows/ci.yml
```

Le workflow est automatiquement déclenché :

- à chaque push sur `main` ;
- lors des pull requests vers `main`.

La CI exécute notamment :

```text
Push GitHub
    ↓
Installation Python
    ↓
Installation des dépendances
    ↓
Tests Pytest
    ↓
Build de l'image Docker
```

Le build Docker ne se lance que si les tests sont passés avec succès.

L'état du pipeline peut être consulté dans l'onglet `Actions` du repository GitHub.

---

# 14. Historique Git

Le projet a été construit progressivement et chaque étape importante fait l'objet d'un commit dédié.

Exemples :

```text
Add trained model artifact
Add initial project files
Add initial FastAPI application
Add API and model unit tests
Add Docker containerization
Add prediction monitoring and production logging
Add monitoring dashboard and data drift analysis
Add CI pipeline with tests and Docker build
```

L'historique complet est disponible dans la liste des commits GitHub.

---

# 15. Explicabilité du modèle

Le modèle LightGBM peut également être expliqué localement grâce à SHAP.

L'objectif est de comprendre, pour une prédiction individuelle, quelles variables ont le plus contribué au score obtenu.

Une explication locale peut notamment inclure :

- le nom de la feature ;
- sa valeur pour le client ;
- sa valeur SHAP ;
- son importance locale.

Un graphique SHAP de type waterfall peut également être utilisé dans le dashboard pour visualiser les contributions positives et négatives des différentes variables.

---

# 16. Technologies utilisées

```text
Python 3.12
FastAPI
Uvicorn
LightGBM
scikit-learn
pandas
NumPy
SHAP
Streamlit
Plotly
SciPy
SQLite
Pytest
Docker
GitHub Actions
uv
```

---

# 17. Lancement rapide

## API locale

```bash
uv sync

uv run uvicorn app.main:app \
    --app-dir src \
    --reload
```

Puis ouvrir :

```text
http://127.0.0.1:8000/docs
```

## Tests

```bash
uv run pytest -v
```

## Docker

```bash
docker build -t p8-credit-api .

docker run --rm \
    -p 8000:8000 \
    p8-credit-api
```

## Monitoring

```bash
uv run streamlit run monitoring/dashboard.py
```

---

# Repository

GitHub :

```text
https://github.com/PaulLARRIBET/P8
```

---

# Auteur

Paul Larribet