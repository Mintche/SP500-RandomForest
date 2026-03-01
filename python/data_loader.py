import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, List

class SP500Loader:
    def __init__(self, start_date: str, end_date: str):
        self.ticker = "^GSPC" # S&P 500
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self) -> pd.DataFrame:
        print(f"Téléchargement des données {self.ticker}...")
        # Téléchargement des données via yfinance
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        return df

    def prepare_features_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if df.empty:
            return np.array([]), np.array([])

        data = df.copy()

        # --- Feature Engineering ---
        # 1. Rendements journaliers
        data['Returns'] = data['Close'].pct_change()
        # 2. Moyennes mobiles (SMA)
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        # 3. Volatilité (Ecart-type glissant)
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        # 4. RSI (Relative Strength Index) - 14 jours
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # 5. MACD (Moving Average Convergence Divergence)
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # 6. Lag Features (Prix des jours précédents)
        data['Lag_1'] = data['Close'].shift(1)
        data['Lag_2'] = data['Close'].shift(2)

        # --- Target ---
        data['Target'] = data['Returns'].shift(-1)

        # Suppression des NaN générés par les rolling windows et le shift
        data.dropna(inplace=True)

        # Sélection des features et conversion en numpy
        features = ['Close', 'Returns', 'SMA_5', 'SMA_20', 'Volatility', 'RSI', 'MACD', 'Signal_Line', 'Lag_1', 'Lag_2']
        
        X = data[features].values
        y = data['Target'].values

        return X, y