import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class CryptoDataCollector:
    def __init__(self):
        self.symbols = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD']

    def fetch_historical_data(self, symbol, period='2y'):
        try:
            crypto_data = yf.download(symbol, period=period, interval='1d')
            return crypto_data
        except Exception as e:
            print(f"Error Fetching Data For {symbol}: {e}")
            return None
    
    def prepare_features(self,df):
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])

        return df
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100/(1 +rs))

    def calculate_macd(self, prices, fast=12, slow=26):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        return exp1 - exp2
        