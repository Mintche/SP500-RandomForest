import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple

class SP500Loader:
    def __init__(self, start_date: str, end_date: str, forecast_horizon: int = 5):
        self.ticker = "^GSPC" # S&P 500
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_horizon = forecast_horizon # Nouvel attribut : Horizon de prédiction (ex: 5 jours)

    def fetch_data(self) -> pd.DataFrame:
        print(f"Téléchargement des données {self.ticker}...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        return df

    def prepare_features_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if df.empty:
            return np.array([]), np.array([])

        data = df.copy()

        # --- Feature Engineering (Ne change pas) ---
        data['Returns'] = data['Close'].pct_change()
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        data['Lag_1'] = data['Close'].shift(1)
        data['Lag_2'] = data['Close'].shift(2)

        # --- NOUVELLE TARGET ---
        # On calcule le rendement cumulé entre aujourd'hui et dans 'forecast_horizon' jours
        # Formule : (Prix_Futur - Prix_Actuel) / Prix_Actuel
        data['Target'] = (data['Close'].shift(-self.forecast_horizon) - data['Close']) / data['Close']

        # Suppression des NaN générés par les rolling windows et le shift futur
        data.dropna(inplace=True)

        features = ['Close', 'Returns', 'SMA_5', 'SMA_20', 'Volatility', 'RSI', 'MACD', 'Signal_Line', 'Lag_1', 'Lag_2']
        
        X = data[features].values
        y = data['Target'].values

        return X, y