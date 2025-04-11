from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, posexplode, concat_ws, concat, lit
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType
import pandas as pd
import os

spark = SparkSession.builder \
    .appName("KafkaStreamToPostgresAndCSV") \
    .master("local[*]") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
            "org.postgresql:postgresql:42.2.18") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

kafka_bootstrap_servers = "localhost:9092"
kafka_topic = "test1"

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic) \
    .option("startingOffsets", "latest") \
    .load()

json_schema = StructType([
    StructField("times", StringType(), True),
    StructField("data", StringType(), True),
    StructField("anomaly_type", StringType(), True),
    StructField("anomaly_times", StringType(), True)
])

def write_to_postgres(batch_df, batch_id, table_name):
    postgres_url = "jdbc:postgresql://localhost:5432/suraj"
    postgres_props = {
        "user": "postgres",
        "password": "Tranzmeo1@#",
        "driver": "org.postgresql.Driver"
    }
    batch_df.write.jdbc(
        url=postgres_url,
        table=table_name,
        mode="append",
        properties=postgres_props
    )

def write_to_csv_single(batch_df, batch_id, path, filename_prefix):
    pandas_df = batch_df.toPandas()
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{filename_prefix}_batch_{batch_id}.csv")
    pandas_df.to_csv(filename, index=False)

def foreach_batch_function(df, epoch_id):
    parsed_df = df.withColumn("json", from_json(col("value").cast("string"), json_schema)).select("json.*")

    parsed_df = parsed_df \
        .withColumn("times", from_json(col("times"), ArrayType(StringType()))) \
        .withColumn("data", from_json(col("data"), ArrayType(ArrayType(DoubleType())))) \
        .withColumn("anomaly_times", from_json(col("anomaly_times"), ArrayType(StringType())))

    data_exploded = parsed_df.selectExpr("posexplode(times) as (idx, time)", "data")
    final_data = data_exploded.select(
        col("time"),
        col("data").getItem(col("idx")).alias("data_array")
    ).withColumn(
        "data",
        concat(lit("{["), concat_ws(",", col("data_array")), lit("]}"))
    ).select("time", "data")

    anomaly_rows = []
    parsed_records = parsed_df.select("anomaly_type", "anomaly_times").collect()

    for row in parsed_records:
        anomaly_type = row["anomaly_type"]
        anomaly_times = row["anomaly_times"]

        if anomaly_times:
            for atime in anomaly_times:
                anomaly_rows.append((atime, anomaly_type))
        else:
            anomaly_rows.append(("", anomaly_type))

    if anomaly_rows:
        anomaly_df = spark.createDataFrame(anomaly_rows, ["anomaly_time", "anomaly_type"])
    else:
        anomaly_df = spark.createDataFrame([], "anomaly_time STRING, anomaly_type STRING")

    write_to_postgres(final_data, epoch_id, "wd")
    write_to_postgres(anomaly_df, epoch_id, "ad")

    write_to_csv_single(final_data, epoch_id, "/home/kannan/Desktop/task/kafka/streamed_output_3/data", "data")
    write_to_csv_single(anomaly_df, epoch_id, "/home/kannan/Desktop/task/kafka/streamed_output_3/anomaly", "anomaly")

query = df.writeStream \
    .foreachBatch(foreach_batch_function) \
    .outputMode("append") \
    .start()

query.awaitTermination()
