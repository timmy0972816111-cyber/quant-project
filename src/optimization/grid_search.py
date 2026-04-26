from itertools import product
from pathlib import Path
import pandas as pd

from src.utils.config import load_yaml
from src.data.fetchers import fetch_yfinance_data
from src.backtest.engine import run_backtest
from src.backtest.metrics import calculate_basic_metrics
from src.strategies.ma_cross import MACrossStrategy


def make_range(start: int, end: int, step: int) -> list[int]:
    return list(range(start, end + 1, step))


def run_ma_cross_grid_search(config_path: str = "config/optimization.yaml") -> tuple[pd.DataFrame, dict]:
    config = load_yaml(config_path)

    symbol = config["data"]["symbol"]
    start = config["data"]["start"]
    end = config["data"]["end"]
    interval = config["data"]["interval"]

    fee_rate = config["backtest"]["fee_rate"]

    short_cfg = config["optimization"]["short_window"]
    long_cfg = config["optimization"]["long_window"]

    metric_name = config["objective"]["metric"]
    ascending = config["objective"]["ascending"]

    short_values = make_range(short_cfg["start"], short_cfg["end"], short_cfg["step"])
    long_values = make_range(long_cfg["start"], long_cfg["end"], long_cfg["step"])

    df = fetch_yfinance_data(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval
    )

    results = []

    for short_window, long_window in product(short_values, long_values):
        # 避免短均線 >= 長均線
        if short_window >= long_window:
            continue

        strategy = MACrossStrategy(
            short_window=short_window,
            long_window=long_window
        )

        signal_df = strategy.generate_signals(df)
        backtest_df = run_backtest(signal_df, fee_rate=fee_rate)
        metrics = calculate_basic_metrics(backtest_df)

        result_row = {
            "symbol": symbol,
            "short_window": short_window,
            "long_window": long_window,
            "total_return": metrics["total_return"],
            "num_days": metrics["num_days"],
            "avg_daily_return": metrics["avg_daily_return"],
            "daily_volatility": metrics["daily_volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
        }

        results.append(result_row)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        raise ValueError("沒有可用的參數組合，請檢查 optimization.yaml")

    result_df = result_df.sort_values(
        by=metric_name,
        ascending=ascending
    ).reset_index(drop=True)

    best_params = result_df.iloc[0].to_dict()

    output_path = Path("data/processed/ma_cross_optimization_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return result_df, best_params