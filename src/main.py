from data.crypto_data_collector import CryptoDataCollector
from models.crypto_predictor import CryptoPredictor
import numpy as np

def main():
    collector = CryptoDataCollector()
    predictor = CryptoPredictor()
    
    btc_data = collector.fetch_historical_data('BTC-USD')
    if btc_data is not None:
        data = collector.prepare_features(btc_data)
        
        closing_prices = data['Close'].values.reshape(-1, 1)
        scaled_data = predictor.scaler.fit_transform(closing_prices)
        
        X, y = predictor.prepare_sequences(scaled_data)
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        predictor.model = predictor.build_model((X_train.shape[1], 1))
        predictor.train(X_train, y_train)

        predictions = predictor.predict(X_test)
        
        predictions = predictor.scaler.inverse_transform(predictions)
        actual = predictor.scaler.inverse_transform(y_test)
        
        print("Prediksi selesai!")
        print(f"RMSE: {np.sqrt(np.mean((predictions - actual) ** 2))}")

        return actual.flatten(), predictions.flatten()
    return None, None
if __name__ == "__main__":
    main()