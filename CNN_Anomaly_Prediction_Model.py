import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential# type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.utils import to_categorical# type: ignore
import psycopg2
from psycopg2 import sql

def get_data(db_params,table_name):
    conn=psycopg2.connect(**db_params)
    cursor=conn.cursor()
    select_query=f"select * from {table_name};"
    df=pd.read_sql(select_query,conn)
    conn.close()
    df=df.drop('Unnamed: 0',axis=1)
    return df

def preprocess_data(df):
    X=df.iloc[:,:100]
    y=df.iloc[:,-1]
    scaler=MinMaxScaler()
    X=scaler.fit_transform(X)
    en=LabelEncoder()
    y=en.fit_transform(y)
    y=to_categorical(y)
    X=X.reshape(X.shape[0],X.shape[1],1)
    return train_test_split(X,y,test_size=0.2,random_state=1)

def build_cnn(input_shape,num_classes):
    model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') 
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def train_cnn(db_params,table_name,epochs=20,batch_size=32):
    df=get_data(db_params,table_name)
    X_train,X_test,y_train,y_test=preprocess_data(df)
    model=build_cnn(input_shape=(X_train.shape[1],1),num_classes=y_train.shape[1])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    loss,accuracy=model.evaluate(X_test,y_test)
    print(accuracy*100)
    return model

db_params={'dbname':'suraj',
                      'user':'postgres',
                      'password':'Tranzmeo1@#',
                      'host':'localhost',
                      'port':'5432'
}
model=train_cnn(db_params,table_name='transposed_df')
    
    
    
