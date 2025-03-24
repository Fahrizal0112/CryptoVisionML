import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class CryptoPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.seq_length = 3  # Tetap gunakan sequence pendek karena performanya bagus
    
    def prepare_sequences(self, data):
        """
        Menyiapkan data sequence
        """
        print("   - Normalizing data...")
        scaled_data = self.scaler.fit_transform(data)
        
        print("   - Creating sequences...")
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i:(i + self.seq_length)])
            y.append(scaled_data[i + self.seq_length])
            
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X untuk LSTM (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"   ✓ Created sequences with shape: X={X.shape}, y={y.shape}")
        return X, y
    
    def build_model(self, input_shape):
        """
        Gunakan model yang sama karena performanya bagus
        """
        model = Sequential([
            LSTM(64, input_shape=input_shape),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Training dengan early stopping
        """
        from tensorflow.keras.callbacks import EarlyStopping
        
        callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        print("Training model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[callback],
            verbose=1
        )
        
        return X_train, y_train  # Return data training untuk digunakan kemudian
    
    def predict(self, X):
        """
        Membuat prediksi
        """
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)

    def save_model(self, path='saved_model'):
        pass

    def load_model(self, path='saved_model'):
        pass
        