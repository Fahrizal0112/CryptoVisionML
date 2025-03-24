from data.crypto_data_collector import CryptoDataCollector
from models.crypto_predictor import CryptoPredictor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def evaluate_predictions(actual, predictions):
    """
    Evaluasi hasil prediksi
    """
    mse = np.mean((predictions - actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actual))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    
    print("\n=== Hasil Evaluasi Model ===")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return {'rmse': rmse, 'mae': mae, 'mape': mape}

def load_data():
    """
    Mengambil data Bitcoin
    """
    print("   - Mengunduh data BTC-USD...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    btc = yf.download('BTC-USD', 
                      start=start_date, 
                      end=end_date,
                      progress=False)
    
    print(f"   ✓ Berhasil mengunduh {len(btc)} data points")
    return btc

def prepare_model_data(df):
    """
    Menyiapkan data untuk model
    """
    # Fokus pada fitur utama
    features = pd.DataFrame({
        'Close': df['Close'],
        'Volume': df['Volume'],
        'High': df['High'],
        'Low': df['Low']
    })
    
    # Normalisasi data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_data, columns=features.columns)
    
    # Buat sequences
    seq_length = 14
    X, y = [], []
    
    for i in range(len(scaled_df) - seq_length):
        X.append(scaled_df.iloc[i:(i + seq_length)].values)
        y.append(scaled_df.iloc[i + seq_length]['Close'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   ✓ Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler

def create_sequences(data, seq_length):
    """
    Buat sequences dengan overlap yang lebih besar
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])  # Ambil harga Close saja
    return np.array(X), np.array(y)

def calculate_atr(df, period=14):
    """
    Menghitung Average True Range
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(period).mean()

def main():
    try:
        print("\nMemulai analisis cryptocurrency...\n")
        
        # Load data
        print("1. Mengambil data...")
        df = load_data()
        
        if df is None or df.empty:
            raise ValueError("Tidak bisa mendapatkan data cryptocurrency")
        
        # Prepare data
        print("\n2. Mempersiapkan data...")
        close_prices = df['Close'].values.reshape(-1, 1)
        
        # Initialize predictor
        predictor = CryptoPredictor()
        
        # Prepare sequences
        X, y = predictor.prepare_sequences(close_prices)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Build and train model
        print("\n3. Melatih model...")
        predictor.build_model((predictor.seq_length, 1))
        predictor.train(X_train, y_train)
        
        # Make predictions
        print("\n4. Melakukan prediksi...")
        y_pred = predictor.model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = predictor.scaler.inverse_transform(y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
        print(f"\nHasil MAPE: {mape:.2f}%")
        
        # Save predictions
        results = pd.DataFrame({
            'Actual': y_test_actual.flatten(),
            'Predicted': y_pred_actual.flatten()
        })
        results.to_csv('predictions.csv', index=False)
        print("\nHasil prediksi disimpan di 'predictions.csv'")
        
        print("\nSelesai!")
        return True
        
    except Exception as e:
        print(f"\nTerjadi error dalam program utama:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return False

if __name__ == "__main__":
    main()