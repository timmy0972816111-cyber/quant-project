import pandas as pd
from src.strategies.base import BaseStrategy


class MACrossStrategy(BaseStrategy):
    """
    均線交叉策略：
    - 短均線 > 長均線 -> signal = 1
    - 否則 -> signal = 0
    """

    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(name="ma_cross")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        df = data.copy()

        df["ma_short"] = df["close"].rolling(self.short_window).mean()
        df["ma_long"] = df["close"].rolling(self.long_window).mean()

        df["signal"] = (df["ma_short"] > df["ma_long"]).astype(int)

        return df