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
        pass

    def prepare_features_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        pass