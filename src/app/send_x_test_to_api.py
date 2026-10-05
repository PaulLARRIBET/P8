import pandas as pd
import requests


API_URL = "http://127.0.0.1:8000/predict"
X_TEST_PATH = "data/x_test.csv"

df = pd.read_csv(X_TEST_PATH)

success = 0
errors = 0

for _, row in df.iterrows():

    payload = {
        key: float(value)
        for key, value in row.to_dict().items()
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            success += 1
        else:
            errors += 1
            print(
                response.status_code,
                response.text
            )

    except requests.RequestException as e:
        errors += 1
        print("Erreur :", e)


print("Succès :", success)
print("Erreurs :", errors)