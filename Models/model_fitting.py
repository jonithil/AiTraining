
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from data_fetching import fetch_data



def normalize(x, norm_type):
    

    if norm_type == 'Minmax':
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
    elif norm_type == 'Standard':
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    return x


def logistic(x_train, x_test, y_train, y_test):

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)



def convol_neural(df):
    #df = fetch_data()
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

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(x_test), axis=1)

    y_pred_labels = label_encoder.inverse_transform(y_pred)

    for i in range(5):
        print(f"Actual: {label_encoder.inverse_transform([y_test[i]])[0]}, Predicted: {y_pred_labels[i]}")




def recurrent(df):
    #df = fetch_data()

    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    y = y.reshape(-1, 1)

    x = normalize(X, 'Minmax')

    x = x.reshape((x.shape[0], x.shape[1], 1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    #model.add(Dense(1, activation='sigmoid'))  # Ensure single neuron for binary classification and sigmoid activaton function

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(x_test), axis=1)

    y_pred_labels = label_encoder.inverse_transform(y_pred)

    for i in range(5):
        print(f"Actual: {label_encoder.inverse_transform([y_test[i]])[0]}, Predicted: {y_pred_labels[i]}")

    # Predict and convert probabilities to binary values
    #y_pred_prob = model.predict(x_test)
    #y_pred = (y_pred_prob > 0.5).astype(int)  # Convert to binary labels (0 or 1)

    # Inverse transform predictions
    #y_pred_labels = label_encoder.inverse_transform(y_pred.flatten())
    #y_test_actual = label_encoder.inverse_transform(y_test.flatten())

    # Print some sample predictions
    #for i in range(5):
        #print(f"Actual: {y_test_actual[i]}, Predicted: {y_pred_labels[i]}")



