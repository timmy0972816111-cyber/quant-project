from __future__ import annotations

from src.utils.config import load_yaml
from src.data.fetchers import fetch_yfinance_data
from src.strategies.ma_cross import MACrossStrategy
from src.execution.paper_broker import PaperBroker
from src.portfolio.position_manager import PositionManager
from src.portfolio.risk_manager import RiskManager


def latest_signal_to_order(latest_signal: int, current_position: int):
    """
    最簡化 signal -> order 規則：
    - signal = 1 且目前空手 -> BUY 1
    - signal = 0 且目前有部位 -> SELL 全部
    """
    if latest_signal == 1 and current_position == 0:
        return {"side": "BUY", "quantity": 1}
    elif latest_signal == 0 and current_position > 0:
        return {"side": "SELL", "quantity": current_position}
    return None


def main():
    # 1. 讀設定
    config = load_yaml("config/strategy.yaml")

    symbol = config["data"]["symbol"]
    start = config["data"]["start"]
    end = config["data"]["end"]
    interval = config["data"]["interval"]

    short_window = config["strategy"]["short_window"]
    long_window = config["strategy"]["long_window"]

    # 2. 初始化 paper broker / manager
    broker = PaperBroker(initial_cash=1_000_000)
    broker.connect()

    position_manager = PositionManager(broker)
    risk_manager = RiskManager(max_order_qty=1, allow_short=False)

    # 3. 抓資料
    df = fetch_yfinance_data(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
    )

    # 4. 跑策略
    strategy = MACrossStrategy(
        short_window=short_window,
        long_window=long_window,
    )

    signal_df = strategy.generate_signals(df)

    # 5. 取最新訊號
    latest_row = signal_df.iloc[-1]
    latest_signal = int(latest_row["signal"])
    latest_close = float(latest_row["close"])

    # 6. 查目前持倉
    current_position = position_manager.get_position(symbol)

    print("========== PAPER TRADING STATUS ==========")
    print(f"symbol           : {symbol}")
    print(f"latest datetime  : {latest_row['datetime']}")
    print(f"latest close     : {latest_close}")
    print(f"latest signal    : {latest_signal}")
    print(f"current position : {current_position}")
    print(f"account info     : {broker.get_account_info()}")

    # 7. 訊號轉訂單
    order_request = latest_signal_to_order(latest_signal, current_position)

    if order_request is None:
        print("No action. 訊號與當前部位一致，不需下單。")
    else:
        side = order_request["side"]
        quantity = order_request["quantity"]

        # 8. 風控檢查
        risk_manager.check_order(
            side=side,
            quantity=quantity,
            current_position=current_position,
        )

        # 9. 送 paper order
        order = broker.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="MKT",
            price=latest_close,   # paper 先用最新 close 當成交價
        )

        print("Order sent:")
        print(order)
        print("Updated positions:", broker.get_positions())
        print("Updated account  :", broker.get_account_info())

    broker.disconnect()


if __name__ == "__main__":
    main()