import json
from kafka import KafkaConsumer
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, TimestampType
from pyspark.sql.functions import col
from datetime import datetime
from threading import Thread
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from pyspark.sql.window import Window
from pyspark.sql.functions import last, first, col
import configparser

config = configparser.ConfigParser()
config.read(r'/home/tranzmeo/Learning/model creation/AiTraining/config.ini')


KAFKA_BROKER = config['Kafka']['KafkaBroker']
TOPIC_NAME = config['Kafka']['TopicName']


DB_URL = config['WellDatabase']['DB_URL']
DB_PROPERTIES = {
   "user": config['WellDatabase']['user'],
   "password": config['WellDatabase']['password'],
   "driver": config['WellDatabase']['driver']
}


RAW_DATA_TABLE = config['WellDatabase']['RAW_DATA_TABLE']

ANOMALY_TABLE = config['WellDatabase']['ANOMALY_TABLE']


spark = SparkSession.builder \
    .appName("KafkaToPostgres") \
    .config("spark.jars", "/home/tranzmeo/jars/postgresql-42.2.27.jar") \
    .getOrCreate()

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset='latest',
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode('utf-8')) if v else None
)

anomaly_schema = StructType([
    StructField("anomaly_time", TimestampType(), True),
    StructField("anomaly_type", StringType(), True)
])


def write_to_postgres(df, table_name):
    df.write \
        .format("jdbc") \
        .option("url", DB_URL) \
        .option("dbtable", table_name) \
        .options(**DB_PROPERTIES) \
        .mode("append") \
        .save()


print(f"Subscribed to topic: {TOPIC_NAME}")

for message in consumer:
    kafka_data = message.value
    event_time = kafka_data.get('times', None)
    data_list = kafka_data.get('line_data', [])
    anomaly_time = kafka_data.get('anomaly_times')
    anomaly_type = kafka_data.get('anomaly_type', None)

    if data_list:
       
        #flat_list = [item for sublist in data_list for item in sublist]

        json_data = json.dumps(data_list)

        df_data= spark.createDataFrame([(event_time, json_data)], ["time", "data"])


        anomaly_data = [(datetime.strptime(anomaly_time, "%Y-%m-%d %H:%M:%S") if anomaly_time else None, anomaly_type)]
    
        
        df_anomaly = spark.createDataFrame(anomaly_data, schema=anomaly_schema)

        df_anomaly = df_anomaly.withColumn("anomaly_time", col("anomaly_time").cast(TimestampType()))

        thread1 = Thread(target=write_to_postgres, args=(df_data, RAW_DATA_TABLE))
        thread2 = Thread(target=write_to_postgres, args=(df_anomaly, ANOMALY_TABLE))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        print("Inserted into PostgreSQL: Both `rawdata` and `anomaly` tables")
