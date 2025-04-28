import pandas as pd
from sqlalchemy import create_engine

def data_upload(signal):

    DATABASE_URL = "postgresql://postgres:tranzmeo@localhost:5432/signal"

    engine = create_engine(DATABASE_URL)

    #df = pd.read_csv(r"/home/tranzmeo/Documents/Models/signal.csv")

    signal.to_sql("signal", engine, if_exists="replace", index=False)

    print("CSV data successfully uploaded to PostgreSQL!")



