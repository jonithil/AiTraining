from pyspark.sql import SparkSession
import configparser
from pyspark.sql.functions import col, size, regexp_replace, split
import pandas as pd



config = configparser.ConfigParser()
config.read(r'/home/tranzmeo/Learning/model_creation/AiTraining/config.ini')


spark = SparkSession.builder \
    .appName("DB Filter Columns") \
    .config("spark.jars", "/home/tranzmeo/jars/postgresql-42.2.27.jar") \
    .getOrCreate()


jdbc_url = config['WellDatabase']['DB_URL']
properties = {
   "user": config['WellDatabase']['user'],
   "password": config['WellDatabase']['password'],
   "driver": config['WellDatabase']['driver']
}


table_name = config['WellDatabase']['RAW_DATA_TABLE']

query = f"(SELECT data FROM {table_name} LIMIT 100) AS subquery"

def data_prep():

    df = spark.read.jdbc(url=jdbc_url, table=query, properties=properties)

    df_array = df.withColumn("data_array", split(regexp_replace("data", r"[\[\]\s]", ""), ","))

    df_with_length = df_array.withColumn("array_length", size(col("data_array")))

    distinct_lengths = df_with_length.select("array_length").distinct().collect()

    lengths = [row["array_length"] for row in distinct_lengths]

    if len(lengths) == 1:
        print(f"All arrays have the same length: {lengths[0]}")
    else:
        print("Not same length")


    data = df_array.select("data_array").rdd.map(lambda row: row["data_array"]).collect()

    selected_df = pd.DataFrame(data)
    selected_df = selected_df.astype(float)

    first_row = selected_df.iloc[0]

    columns_to_keep = [
        col for col, val in first_row.items()
        if isinstance(val, (int, float)) and val > 335
    ]

    filtered_pdf = selected_df[columns_to_keep]

    transposed = filtered_pdf.T

    return transposed