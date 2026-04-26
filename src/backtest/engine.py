import numpy as np
import pandas as pd


def run_backtest(
    data: pd.DataFrame,
    fee_rate: float = 0.001,
    allow_short: bool = True,
) -> pd.DataFrame:
    """
    通用狀態機回測器：
    不負責計算 ATR / 指標，只負責執行策略輸出的欄位。

    必要欄位：
    - datetime, open, high, low, close
    - entry_long, exit_long
    - entry_short, exit_short
    - long_stop_price, short_stop_price
    """

    required_cols = [
        "datetime", "open", "high", "low", "close",
        "entry_long", "exit_long",
        "entry_short", "exit_short",
        "long_stop_price", "short_stop_price",
    ]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"資料缺少必要欄位: {missing_cols}")

    df = data.copy().sort_values("datetime").reset_index(drop=True)

    df["position"] = 0
    df["trade"] = 0
    df["entry_price"] = np.nan
    df["exit_price"] = np.nan
    df["active_stop_price"] = np.nan
    df["signal"] = 0

    df["market_return"] = df["close"].pct_change().fillna(0)
    df["strategy_return_before_cost"] = 0.0
    df["transaction_cost"] = 0.0
    df["strategy_return"] = 0.0

    current_pos = 0

    for i in range(len(df)):
        row = df.loc[i]

        # 用前一根持倉吃這一根報酬
        if i > 0:
            prev_pos = df.loc[i - 1, "position"]
            market_ret = df.loc[i, "market_return"]
            df.loc[i, "strategy_return_before_cost"] = prev_pos * market_ret
        else:
            prev_pos = 0
            df.loc[i, "strategy_return_before_cost"] = 0.0

        new_pos = current_pos

        # ===== 先判斷出場 =====
        if current_pos == 1:
            stop_hit = pd.notna(row["long_stop_price"]) and row["low"] <= row["long_stop_price"]
            if bool(row["exit_long"]) or stop_hit:
                new_pos = 0
                df.loc[i, "exit_price"] = row["close"]

        elif current_pos == -1:
            stop_hit = pd.notna(row["short_stop_price"]) and row["high"] >= row["short_stop_price"]
            if bool(row["exit_short"]) or stop_hit:
                new_pos = 0
                df.loc[i, "exit_price"] = row["close"]

        # ===== 平倉後才看進場 =====
        if new_pos == 0:
            if bool(row["entry_long"]):
                new_pos = 1
                df.loc[i, "entry_price"] = row["close"]
                df.loc[i, "signal"] = 1

            elif allow_short and bool(row["entry_short"]):
                new_pos = -1
                df.loc[i, "entry_price"] = row["close"]
                df.loc[i, "signal"] = -1

        # ===== 目前生效的停損價 =====
        if new_pos == 1:
            df.loc[i, "active_stop_price"] = row["long_stop_price"]
        elif new_pos == -1:
            df.loc[i, "active_stop_price"] = row["short_stop_price"]
        else:
            df.loc[i, "active_stop_price"] = np.nan

        # ===== 交易成本 =====
        trade_size = abs(new_pos - current_pos)
        df.loc[i, "trade"] = trade_size
        df.loc[i, "transaction_cost"] = trade_size * fee_rate
        df.loc[i, "strategy_return"] = (
            df.loc[i, "strategy_return_before_cost"] - df.loc[i, "transaction_cost"]
        )

        df.loc[i, "position"] = new_pos
        current_pos = new_pos

    df["equity_curve"] = (1 + df["strategy_return"].fillna(0)).cumprod()

    return df