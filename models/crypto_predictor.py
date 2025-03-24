import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class CryptoPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()

    def prepare_sequences(self, data, seq_length=60):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i+seq_length)])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1
        )

    def predict(self, X):
        return self.model.predict(X)
        