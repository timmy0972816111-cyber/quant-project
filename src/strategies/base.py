from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """
    所有策略的基底類別
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        根據輸入資料產生交易訊號
        必須回傳一個包含訊號欄位的 DataFrame
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        檢查策略所需欄位是否存在
        """
        required_cols = ["datetime", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            raise ValueError(f"資料缺少必要欄位: {missing_cols}")