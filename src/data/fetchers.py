import pandas as pd
import yfinance as yf


def fetch_yfinance_data(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    從 Yahoo Finance 抓資料，回傳標準化後的 DataFrame
    """
    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"抓不到資料: {symbol}")

    # 若欄位是 MultiIndex，攤平成單層
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    df = df.reset_index()

    # 統一欄位名稱
    rename_map = {
        "Date": "datetime",
        "Datetime": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # 加入 symbol 欄位
    df["symbol"] = symbol

    # 欄位排序
    base_cols = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
    optional_cols = [c for c in ["adj_close"] if c in df.columns]
    final_cols = [c for c in base_cols if c in df.columns] + optional_cols

    df = df[final_cols]

    return df