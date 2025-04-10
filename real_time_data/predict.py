from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType
from datetime import datetime
import pandas as pd
import requests
import json
from fetch import data_prep
from joblib import load
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import configparser



def predict_from_model():
    test_input = data_prep()
    original_index = test_input.index.to_list()
    reshaped_input = test_input.values.reshape((test_input.shape[0], test_input.shape[1], 1))

    payload = {
        "instances": reshaped_input.tolist()
    }

    response = requests.post(
        "http://localhost:8501/v1/models/cnn_model:predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        raise Exception(f"Prediction request failed: {response.text}")

    pred_probs = np.array(response.json()["predictions"])
    pred_classes = np.argmax(pred_probs, axis=1)

    label_encoder = load("label_encoder.pkl")
    decoded_labels = label_encoder.inverse_transform(pred_classes)

    result_df = pd.DataFrame({
        'index': original_index,
        'prediction': decoded_labels,
        'timestamp': [datetime.now()] * len(decoded_labels)
    })
    
    return result_df

def store_predictions(result_df):

    config = configparser.ConfigParser()
    config.read('/home/tranzmeo/Learning/model_creation/AiTraining/config.ini')

    spark = SparkSession.builder \
        .appName("Store Predictions") \
        .config("spark.jars", "/path/to/postgresql.jar").getOrCreate()
        

    spark_df = spark.createDataFrame(result_df)

    jdbc_url = config['WellDatabase']['DB_URL']
    properties = {
        "user": config['WellDatabase']['user'],
        "password": config['WellDatabase']['password'],
        "driver": config['WellDatabase']['driver']
    }

    
    spark_df.write.jdbc(
        url=jdbc_url,
        table="prediction_results",
        mode="append",
        properties=properties
    )

    print("Predictions stored in the database.")




if __name__ == "__main__":
    pred_df = predict_from_model()
    store_predictions(pred_df)

