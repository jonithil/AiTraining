from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, avg
from pyspark.sql.types import StructType, ArrayType, DoubleType
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
kafka_topic = "docker_data"

json_schema = StructType().add("line_data", ArrayType(DoubleType()))

def pad_or_truncate(lst, target_length):
    if len(lst) < target_length:
        return lst + [None] * (target_length - len(lst))
    else:
        return lst[:target_length]

def write_to_postgres(df, table_name):
    postgres_url = "jdbc:postgresql://localhost:5432/suraj"
    postgres_props = {
        "user": "postgres",
        "password": "Tranzmeo1@#",
        "driver": "org.postgresql.Driver"
    }
    df.write.jdbc(url=postgres_url, table=table_name, mode="append", properties=postgres_props)

def write_to_csv(df, path, filename):
    os.makedirs(path, exist_ok=True)
    df.toPandas().to_csv(os.path.join(path, filename), index=False)

message_buffer = []

def foreach_batch_function(batch_df, batch_id):
    global message_buffer

    parsed_df = batch_df.withColumn("json", from_json(col("value").cast("string"), json_schema)).select("json.line_data")
    messages = parsed_df.collect()

    for row in messages:
        if row.line_data is not None:
            message_buffer.append(row.line_data)

    print(f"Epoch {batch_id}: Total messages collected = {len(message_buffer)}")

    if len(message_buffer) < 100:
        print(f"Epoch {batch_id}: Waiting for 100 messages.")
        return

    selected_msgs = message_buffer[:100]
    message_buffer = message_buffer[100:]

    max_len = max(len(msg) for msg in selected_msgs)
    uniform_msgs = [pad_or_truncate(msg, max_len) for msg in selected_msgs]

    row_rdd = spark.sparkContext.parallelize(uniform_msgs)
    row_df = spark.createDataFrame(row_rdd.map(lambda x: tuple(x)))
    col_names = [f"col_{i}" for i in range(max_len)]
    final_df = row_df.toDF(*col_names)

    avg_row = final_df.select([avg(col(c)).alias(c) for c in final_df.columns]).collect()[0].asDict()

    print(f"Epoch {batch_id} - Column Averages:")
    for c, val in avg_row.items():
        if val is not None:
            print(f"  {c}: {val:.2f}")

    threshold = 300
    valid_cols = [(col_name, avg_val) for col_name, avg_val in avg_row.items() if avg_val and avg_val > threshold]
    
    top_cols = sorted(valid_cols, key=lambda x: x[1], reverse=True)[:20]
    selected_col_names = [col_name for col_name, _ in top_cols]

    if not selected_col_names:
        print(f"⚠️ Epoch {batch_id}: No valid columns. Skipping.")
        return

    filtered_df = final_df.select(selected_col_names)

    write_to_postgres(filtered_df, "line_data_filtered")
    write_to_csv(filtered_df, "/home/kannan/Desktop/task/kafka/streamed_output_final", f"filtered_epoch_{batch_id}.csv")

    print(f"Epoch {batch_id}: Saved {len(selected_col_names)} columns and 3 rows.")

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic) \
    .option("startingOffsets", "latest") \
    .load()

query = df.writeStream \
    .foreachBatch(foreach_batch_function) \
    .outputMode("append") \
    .start()

query.awaitTermination()
