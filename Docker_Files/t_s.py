import pandas as pd
import numpy as np
import requests
from sqlalchemy import create_engine

db_user = "postgres"
db_password = "Tranzmeo1%40#"
db_host = "localhost"
db_port = "5432"
db_name = "suraj"
table_name = "line_data_filtered"  
engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 100", engine)

feature_col = df.columns[0]

data = df[feature_col].values
reshaped = data.reshape(1, 100, 1)

payload = {
    "instances": reshaped.tolist()
}
url = "http://localhost:8501/v1/models/activity_model:predict"
response = requests.post(url, json=payload)

class_names = {
    0: "manual digging",
    1: "machine digging",
    2: "vehicle movement"
}

if response.status_code == 200:
    probs = response.json()["predictions"][0]
    predicted_class = int(np.argmax(probs))
    predicted_label = class_names.get(predicted_class, "Unknown")

    df["id"] = df.index
    df["prediction"] = predicted_label
    df_result = df[["id", feature_col, "prediction"]]

    print(df_result.head())
else:
    print("Error:", response.text)
