from data.crypto_data_collector import CryptoDataCollector
from models.crypto_predictor import CryptoPredictor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_predictions(actual, predicted, save_path='analysis_results.png'):
    """
    Membuat visualisasi hasil prediksi yang lebih baik
    """
    # Buat figure dengan 2 subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # Set style
    plt.style.use('seaborn')
    
    # Plot harga di subplot pertama
    ax1.plot(actual, label='Harga Aktual', color='blue', linewidth=2)
    ax1.plot(predicted, label='Prediksi', color='red', linestyle='--', linewidth=2)
    
    # Customize plot harga
    ax1.set_title(f'Prediksi Harga Bitcoin (MAPE: {calculate_mape(actual, predicted):.2f}%)', 
                  fontsize=14, pad=20)
    ax1.set_xlabel('Periode', fontsize=12)
    ax1.set_ylabel('Harga (USD)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Tambahkan status harga
    last_actual = actual[-1]
    last_pred = predicted[-1]
    status = "NETRAL"
    if last_pred > last_actual * 1.01:
        status = "BULLISH"
    elif last_pred < last_actual * 0.99:
        status = "BEARISH"
    
    # Tambahkan text status
    ax1.text(0.02, 0.95, f'Status Harga: {status}',
             transform=ax1.transAxes,
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualisasi disimpan di '{save_path}'")
    
    # Close plot untuk menghemat memori
    plt.close()

def calculate_mape(actual, predicted):
    """
    Menghitung MAPE
    """
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_rsi(prices, periods=14):
    """
    Menghitung Relative Strength Index
    """
    deltas = np.diff(prices)
    seed = deltas[:periods+1]
    up = seed[seed >= 0].sum()/periods
    down = -seed[seed < 0].sum()/periods
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:periods] = 100. - 100./(1. + rs)
    
    for i in range(periods, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up*(periods - 1) + upval)/periods
        down = (down*(periods - 1) + downval)/periods
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
        
    return rsi

def main():
    try:
        print("\nMemulai analisis cryptocurrency...\n")
        
        # Load dan prepare data
        df = load_data()
        close_prices = df['Close'].values.reshape(-1, 1)
        
        # Initialize dan train model
        predictor = CryptoPredictor()
        X, y = predictor.prepare_sequences(close_prices)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Train model
        predictor.build_model((predictor.seq_length, 1))
        predictor.train(X_train, y_train)
        
        # Make predictions
        y_pred = predictor.model.predict(X_test, verbose=0)
        
        # Inverse transform
        y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = predictor.scaler.inverse_transform(y_pred)
        
        # Calculate MAPE
        mape = calculate_mape(y_test_actual, y_pred_actual)
        print(f"\nHasil MAPE: {mape:.2f}%")
        
        # Save predictions
        results = pd.DataFrame({
            'Actual': y_test_actual.flatten(),
            'Predicted': y_pred_actual.flatten()
        })
        results.to_csv('predictions.csv', index=False)
        
        # Create visualization
        plot_predictions(
            actual=results['Actual'].values,
            predicted=results['Predicted'].values
        )
        
        print("\nSelesai!")
        return True
        
    except Exception as e:
        print(f"\nTerjadi error:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        return False

if __name__ == "__main__":
    main()