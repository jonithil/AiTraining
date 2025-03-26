import pandas as pd
import psycopg2

def fetch_data():
  conn = psycopg2.connect(
    dbname="signal",
    user="postgres",
    password="tranzmeo",
    host="localhost",
    port="5432"
  )

  df = pd.read_sql("SELECT * FROM signal;", conn)


  conn.close()
  return df
