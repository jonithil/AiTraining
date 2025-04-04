import pandas as pd
import psycopg2

def fetch_data():
  conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="tranzmeo",
    host="localhost",
    port="5432"
  )

  df = pd.read_sql("SELECT * FROM rawdata;", conn)


  conn.close()
  return df



df = fetch_data()
print(df)