from src.utils.config import load_yaml
from src.data.fetchers import fetch_yfinance_data
from src.strategies.ma_cross import MACrossStrategy
from src.backtest.engine import run_backtest
from src.backtest.metrics import calculate_full_metrics
from src.backtest.diagnosis import diagnose_performance

def main():
    config = load_yaml("config/strategy.yaml")

    symbol = config["data"]["symbol"]
    start = config["data"]["start"]
    end = config["data"]["end"]
    interval = config["data"]["interval"]

    short_window = config["strategy"]["short_window"]
    long_window = config["strategy"]["long_window"]

    fee_rate = config["backtest"]["fee_rate"]

    df = fetch_yfinance_data(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval
    )

    strategy = MACrossStrategy(
        short_window=short_window,
        long_window=long_window
    )

    signal_df = strategy.generate_signals(df)
    backtest_df = run_backtest(signal_df, fee_rate=fee_rate)

    metrics, trade_df, worst_dd_df = calculate_full_metrics(
        backtest_df,
        annual_trading_days=252,
        rolling_window=126,
        verbose=True,
        show_plots=True,
    )

    diagnosis_df, avg_score, overall_comment = diagnose_performance(
        metrics,
        verbose=True
    )   


if __name__ == "__main__":
    main()