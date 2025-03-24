import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class CryptoDataCollector:
    def __init__(self):
        self.symbols = ['BTC-USD', 'ETH-USD']

    def fetch_historical_data(self, symbol, period='2y'):
        """
        Mengambil data historis cryptocurrency
        """
        try:
            print(f"\nMengambil data historis untuk {symbol}...")
            crypto_data = yf.download(symbol, period=period, interval='1d')
            
            # Perbaikan untuk MultiIndex
            if isinstance(crypto_data.columns, pd.MultiIndex):
                crypto_data = crypto_data.droplevel(1, axis=1)
            
            print(f"Berhasil mengambil {len(crypto_data)} data points")
            return crypto_data
        except Exception as e:
            print(f"Error mengambil data untuk {symbol}: {e}")
            return None
    
    def prepare_features(self, df):
        """
        Membuat fitur-fitur untuk model prediksi
        """
        print("   - Menghitung technical indicators...")
        
        # Buat copy dari DataFrame asli
        result = df.copy()
        
        try:
            # Simple Moving Averages
            result['SMA_7'] = result['Close'].rolling(window=7).mean()
            result['SMA_30'] = result['Close'].rolling(window=30).mean()
            
            # Bollinger Bands
            sma20 = result['Close'].rolling(window=20).mean()
            std20 = result['Close'].rolling(window=20).std()
            result['BB_middle'] = sma20
            result['BB_upper'] = sma20 + (2 * std20)
            result['BB_lower'] = sma20 - (2 * std20)
            
            # RSI
            result['RSI'] = self.calculate_rsi(result['Close'])
            
            # MACD
            exp1 = result['Close'].ewm(span=12, adjust=False).mean()
            exp2 = result['Close'].ewm(span=26, adjust=False).mean()
            result['MACD'] = exp1 - exp2
            result['Signal_Line'] = result['MACD'].ewm(span=9, adjust=False).mean()
            
            # Price vs Moving Averages
            ma50 = result['Close'].rolling(window=50).mean()
            ma200 = result['Close'].rolling(window=200).mean()
            result['Price_vs_MA50'] = result['Close'] / ma50
            result['Price_vs_MA200'] = result['Close'] / ma200
            
            # Volume indicators
            result['Volume_MA20'] = result['Volume'].rolling(window=20).mean()
            result['Volume_Ratio'] = result['Volume'] / result['Volume_MA20']
            
            # Forward fill NaN values
            result = result.fillna(method='ffill')
            
            print("   ✓ Technical indicators selesai dihitung")
            print(f"   ✓ Total kolom: {len(result.columns)}")
            print("   ✓ Kolom yang tersedia:", list(result.columns))
            
            return result
            
        except Exception as e:
            print(f"Error dalam prepare_features: {str(e)}")
            print("Columns in DataFrame:", df.columns)
            raise e
    
    def calculate_rsi(self, prices, period=14):
        """
        Menghitung Relative Strength Index
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        return exp1 - exp2
        
    def add_price_indicators(self, df):
        """
        Menambahkan indikator untuk analisis harga
        """
        try:
            # Buat DataFrame baru untuk menyimpan indikator
            indicators = pd.DataFrame(index=df.index)
            
            # Bollinger Bands
            indicators['BB_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            indicators['BB_upper'] = indicators['BB_middle'] + (2 * bb_std)
            indicators['BB_lower'] = indicators['BB_middle'] - (2 * bb_std)
            
            # RSI
            indicators['RSI'] = self.calculate_rsi(df['Close'])
            
            # Harga relatif terhadap MA
            indicators['Price_vs_MA50'] = df['Close'] / df['Close'].rolling(window=50).mean()
            indicators['Price_vs_MA200'] = df['Close'] / df['Close'].rolling(window=200).mean()
            
            # Volume analysis
            indicators['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            indicators['Volume_Ratio'] = df['Volume'] / indicators['Volume_MA20']
            
            # Gabungkan DataFrame asli dengan indicators
            result = pd.concat([df, indicators], axis=1)
            return result
            
        except Exception as e:
            print(f"Error in add_price_indicators: {e}")
            return df
    
    def analyze_price_status(self, df):
        """
        Menganalisis status harga (mahal/murah)
        """
        try:
            # Ambil row terakhir yang valid
            last_row = df.iloc[-1]
            
            # Ambil nilai-nilai yang diperlukan
            current_price = float(last_row['Close'])
            current_rsi = float(last_row['RSI'])
            current_bb_upper = float(last_row['BB_upper'])
            current_bb_lower = float(last_row['BB_lower'])
            current_price_ma50 = float(last_row['Price_vs_MA50'])
            current_volume_ratio = float(last_row['Volume_Ratio'])
            
            print(f"\nNilai Current:")
            print(f"Price: ${current_price:,.2f}")
            print(f"RSI: {current_rsi:.2f}")
            print(f"BB Upper: ${current_bb_upper:,.2f}")
            print(f"BB Lower: ${current_bb_lower:,.2f}")
            
            # Inisialisasi skor dan signals
            score = 50
            signals = []
            
            # Analisis RSI
            if current_rsi > 70:
                score += 20
                signals.append(f"RSI ({current_rsi:.2f}) menunjukkan overbought (mahal)")
            elif current_rsi < 30:
                score -= 20
                signals.append(f"RSI ({current_rsi:.2f}) menunjukkan oversold (murah)")
            
            # Analisis Bollinger Bands
            if current_price > current_bb_upper:
                score += 15
                signals.append(f"Harga (${current_price:,.2f}) di atas Bollinger Bands upper (${current_bb_upper:,.2f})")
            elif current_price < current_bb_lower:
                score -= 15
                signals.append(f"Harga (${current_price:,.2f}) di bawah Bollinger Bands lower (${current_bb_lower:,.2f})")
            
            # Analisis MA
            if current_price_ma50 > 1.1:
                score += 10
                signals.append(f"Harga {(current_price_ma50-1)*100:.1f}% di atas MA50")
            elif current_price_ma50 < 0.9:
                score -= 10
                signals.append(f"Harga {(1-current_price_ma50)*100:.1f}% di bawah MA50")
            
            # Volume Analysis
            if current_volume_ratio > 2:
                volume_score = 5 if score > 50 else -5
                score += volume_score
                signals.append(f"Volume {current_volume_ratio:.1f}x lebih tinggi dari rata-rata")
            
            # Status akhir
            if score >= 70:
                status = "MAHAL"
            elif score <= 30:
                status = "MURAH"
            else:
                status = "NETRAL"
            
            # Hitung posisi dalam Bollinger Bands
            bb_range = current_bb_upper - current_bb_lower
            if bb_range > 0:
                bb_position = ((current_price - current_bb_lower) / bb_range) * 100
            else:
                bb_position = 50
            
            return {
                'status': status,
                'score': score,
                'signals': signals,
                'current_price': current_price,
                'rsi': current_rsi,
                'bb_position': bb_position
            }
            
        except Exception as e:
            print(f"\nError dalam analyze_price_status:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            return {
                'status': "ERROR",
                'score': 50,
                'signals': ["Terjadi error dalam analisis"],
                'current_price': 0,
                'rsi': 0,
                'bb_position': 50
            }