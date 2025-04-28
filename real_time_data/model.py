
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



    


def normalize(x, norm_type):
    

    if norm_type == 'Minmax':
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
    elif norm_type == 'Standard':
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    return x
 


def convol_neural(df):
    
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    x = normalize(X, norm_type='Standard')

    x = x.reshape((x.shape[0], x.shape[1], 1))

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
    
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
    
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(np.unique(y)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    EXPORT_PATH = "export/cnn_model/1"
    os.makedirs(EXPORT_PATH, exist_ok=True)
    model.export(EXPORT_PATH)

    joblib.dump(label_encoder, 'label_encoder.pkl')

    