import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Formatting helpers
# -----------------------------
PERCENT_METRICS = {
    "total_return",
    "cagr",
    "vol",
    "mdd",
    "win_rate",
    "lose_rate",
    "avg_win",
    "avg_loss",
    "expectancy",
    "downside_vol",
    "var_95",
    "cvar_95",
    "best_day",
    "worst_day",
    "best_month",
    "worst_month",
    "best_year",
    "worst_year",
    "exposure",
    "daily_turnover",
}


def _format_metric_value(metric_name: str, value) -> str:
    if value is None:
        return "NaN"

    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return "NaN"

    if metric_name in PERCENT_METRICS:
        return f"{value:.2%}"

    if metric_name in {"total_trades", "longest_dd_duration"}:
        return f"{int(value)}"

    if metric_name in {"trades_per_year", "avg_hold_days"}:
        return f"{value:.2f}"

    if isinstance(value, (int, float, np.floating)):
        return f"{value:.3f}"

    return str(value)


# -----------------------------
# Core helpers
# -----------------------------
def _safe_div(a, b, default=np.nan):
    if b is None or b == 0 or pd.isna(b):
        return default
    return a / b


def _extract_trade_df(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """
    從 backtest_df 中抽出交易紀錄。
    目前假設：
    - position 只有 0/1
    - 進場：position 從 0 -> 1
    - 出場：position 從 1 -> 0
    - 報酬使用 equity_curve 變化估算
    """
    df = backtest_df.copy()

    required_cols = ["datetime", "close", "position", "equity_curve"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"_extract_trade_df 缺少必要欄位: {missing_cols}")

    df = df.reset_index(drop=True)

    trades = []
    in_trade = False
    entry_idx = None

    for i in range(len(df)):
        pos = df.loc[i, "position"]
        prev_pos = df.loc[i - 1, "position"] if i > 0 else 0

        if (not in_trade) and prev_pos == 0 and pos == 1:
            in_trade = True
            entry_idx = i

        elif in_trade and prev_pos == 1 and pos == 0:
            exit_idx = i
            entry_row = df.loc[entry_idx]
            exit_row = df.loc[exit_idx]

            entry_equity = df.loc[entry_idx - 1, "equity_curve"] if entry_idx > 0 else 1.0
            exit_equity = df.loc[exit_idx - 1, "equity_curve"] if exit_idx > 0 else df.loc[exit_idx, "equity_curve"]

            trade_return = _safe_div(exit_equity, entry_equity, default=np.nan) - 1
            hold_days = (pd.to_datetime(exit_row["datetime"]) - pd.to_datetime(entry_row["datetime"])).days

            trades.append({
                "entry_time": entry_row["datetime"],
                "exit_time": exit_row["datetime"],
                "entry_price": entry_row["close"],
                "exit_price": exit_row["close"],
                "return": trade_return,
                "hold_days": hold_days,
            })

            in_trade = False
            entry_idx = None

    if in_trade and entry_idx is not None:
        entry_row = df.loc[entry_idx]
        exit_row = df.loc[len(df) - 1]

        entry_equity = df.loc[entry_idx - 1, "equity_curve"] if entry_idx > 0 else 1.0
        exit_equity = df.loc[len(df) - 1, "equity_curve"]

        trade_return = _safe_div(exit_equity, entry_equity, default=np.nan) - 1
        hold_days = (pd.to_datetime(exit_row["datetime"]) - pd.to_datetime(entry_row["datetime"])).days

        trades.append({
            "entry_time": entry_row["datetime"],
            "exit_time": exit_row["datetime"],
            "entry_price": entry_row["close"],
            "exit_price": exit_row["close"],
            "return": trade_return,
            "hold_days": hold_days,
        })

    trade_df = pd.DataFrame(trades)
    return trade_df


def _calc_drawdown_info(equity: pd.Series) -> tuple[pd.Series, int, pd.DataFrame]:
    """
    回傳：
    - drawdown series
    - longest drawdown duration
    - worst drawdown periods dataframe
    """
    peak = equity.cummax()
    drawdown = equity / peak - 1

    dd_flag = drawdown < 0
    longest_dd_duration = 0
    current_dd = 0

    for flag in dd_flag:
        if flag:
            current_dd += 1
            longest_dd_duration = max(longest_dd_duration, current_dd)
        else:
            current_dd = 0

    periods = []
    in_dd = False
    start_idx = None

    for i in range(len(drawdown)):
        if (not in_dd) and drawdown.iloc[i] < 0:
            in_dd = True
            start_idx = i
        elif in_dd and drawdown.iloc[i] == 0:
            end_idx = i
            dd_slice = drawdown.iloc[start_idx:end_idx + 1]
            periods.append({
                "start": drawdown.index[start_idx],
                "end": drawdown.index[end_idx],
                "min_drawdown": dd_slice.min(),
                "duration": len(dd_slice),
            })
            in_dd = False
            start_idx = None

    if in_dd and start_idx is not None:
        dd_slice = drawdown.iloc[start_idx:]
        periods.append({
            "start": drawdown.index[start_idx],
            "end": drawdown.index[-1],
            "min_drawdown": dd_slice.min(),
            "duration": len(dd_slice),
        })

    worst_dd_df = pd.DataFrame(periods)
    if not worst_dd_df.empty:
        worst_dd_df = worst_dd_df.sort_values("min_drawdown").reset_index(drop=True)

    return drawdown, longest_dd_duration, worst_dd_df


# -----------------------------
# Main metrics function
# -----------------------------
def calculate_full_metrics(
    backtest_df: pd.DataFrame,
    annual_trading_days: int = 252,
    rolling_window: int = 126,
    verbose: bool = False,
    show_plots: bool = False,
):
    """
    完整版績效分析。
    必要欄位至少包含：
    - datetime
    - strategy_return
    - equity_curve
    若要更完整 trade/exposure/turnover 分析，建議包含：
    - position
    - trade
    - close
    """
    df = backtest_df.copy()

    required_cols = ["datetime", "strategy_return", "equity_curve"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"calculate_full_metrics 缺少必要欄位: {missing_cols}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")

    returns = df["strategy_return"].fillna(0)
    equity = df["equity_curve"].ffill().fillna(1.0)

    total_return = equity.iloc[-1] - 1

    num_days = len(df)
    years = num_days / annual_trading_days if num_days > 0 else np.nan
    cagr = (equity.iloc[-1] ** (1 / years) - 1) if years and years > 0 else np.nan

    vol = returns.std() * np.sqrt(annual_trading_days)
    sharpe = _safe_div(returns.mean() * np.sqrt(annual_trading_days), returns.std(), default=np.nan)

    drawdown, longest_dd_duration, worst_dd_df = _calc_drawdown_info(equity)
    mdd = drawdown.min()

    if all(c in df.columns for c in ["position", "close"]):
        trade_df = _extract_trade_df(df.reset_index())
    else:
        trade_df = pd.DataFrame(columns=["entry_time", "exit_time", "entry_price", "exit_price", "return", "hold_days"])

    total_trades = len(trade_df)

    if total_trades > 0:
        wins = trade_df[trade_df["return"] > 0]
        losses = trade_df[trade_df["return"] < 0]

        win_rate = len(wins) / total_trades
        lose_rate = len(losses) / total_trades

        avg_win = wins["return"].mean() if len(wins) > 0 else 0.0
        avg_loss = losses["return"].mean() if len(losses) > 0 else 0.0

        payoff_ratio = _safe_div(avg_win, abs(avg_loss), default=np.nan)

        gross_profit = wins["return"].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses["return"].sum()) if len(losses) > 0 else 0.0
        profit_factor = _safe_div(gross_profit, gross_loss, default=np.nan)

        avg_hold_days = trade_df["hold_days"].mean()
        expectancy = trade_df["return"].mean()
        trades_per_year = total_trades / years if years and years > 0 else np.nan
    else:
        win_rate = 0.0
        lose_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        payoff_ratio = np.nan
        profit_factor = np.nan
        avg_hold_days = 0.0
        expectancy = 0.0
        trades_per_year = 0.0

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(annual_trading_days) if len(downside) > 0 else np.nan
    sortino = _safe_div(returns.mean() * np.sqrt(annual_trading_days), downside.std(), default=np.nan)

    calmar = _safe_div(cagr, abs(mdd), default=np.nan)

    pos_ret = returns[returns > 0]
    neg_ret = returns[returns < 0]
    omega = _safe_div(pos_ret.sum(), abs(neg_ret.sum()), default=np.nan)

    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else np.nan

    skewness = returns.skew()
    kurtosis = returns.kurt()

    q95 = returns.quantile(0.95)
    q05 = abs(returns.quantile(0.05))
    tail_ratio = _safe_div(q95, q05, default=np.nan)

    common_sense_ratio = _safe_div(total_return, abs(mdd), default=np.nan)

    if len(equity) > 1 and (equity > 0).all():
        y = np.log(equity.values)
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        stability = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    else:
        stability = np.nan

    exposure = df["position"].abs().mean() if "position" in df.columns else np.nan
    daily_turnover = df["trade"].mean() if "trade" in df.columns else np.nan

    best_day = returns.max()
    worst_day = returns.min()

    monthly_ret = equity.resample("ME").last().pct_change().dropna()
    yearly_ret = equity.resample("YE").last().pct_change().dropna()

    best_month = monthly_ret.max() if len(monthly_ret) > 0 else np.nan
    worst_month = monthly_ret.min() if len(monthly_ret) > 0 else np.nan
    best_year = yearly_ret.max() if len(yearly_ret) > 0 else np.nan
    worst_year = yearly_ret.min() if len(yearly_ret) > 0 else np.nan

    rolling_mean = returns.rolling(rolling_window).mean()
    rolling_std = returns.rolling(rolling_window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(annual_trading_days)
    rolling_vol = rolling_std * np.sqrt(annual_trading_days)

    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "drawdown": drawdown,
        "trade_df": trade_df,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "lose_rate": lose_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "avg_hold_days": avg_hold_days,
        "expectancy": expectancy,
        "trades_per_year": trades_per_year,
        "sortino": sortino,
        "calmar": calmar,
        "omega": omega,
        "downside_vol": downside_vol,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "tail_ratio": tail_ratio,
        "common_sense_ratio": common_sense_ratio,
        "stability": stability,
        "exposure": exposure,
        "daily_turnover": daily_turnover,
        "best_day": best_day,
        "worst_day": worst_day,
        "monthly_ret": monthly_ret,
        "yearly_ret": yearly_ret,
        "best_month": best_month,
        "worst_month": worst_month,
        "best_year": best_year,
        "worst_year": worst_year,
        "rolling_sharpe": rolling_sharpe,
        "rolling_vol": rolling_vol,
        "longest_dd_duration": longest_dd_duration,
        "worst_dd_df": worst_dd_df,
        "equity": equity,
    }

    if verbose:
        print("========== Trading Record ==========")
        print(trade_df)

        print("========== Performance Summary ==========")
        summary_items = {
            "total_return": total_return,
            "cagr": cagr,
            "vol": vol,
            "sharpe": sharpe,
            "mdd": mdd,
        }
        for k, v in summary_items.items():
            print(f"{k:<20}: {_format_metric_value(k, v)}")

        print("\n========== Trade Statistics ==========")
        trade_items = {
            "total_trades": total_trades,
            "trades_per_year": trades_per_year,
            "win_rate": win_rate,
            "lose_rate": lose_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff_ratio,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_hold_days": avg_hold_days,
        }
        for k, v in trade_items.items():
            print(f"{k:<20}: {_format_metric_value(k, v)}")

        print("\n========== Added Metrics ==========")
        extra_items = {
            "sortino": sortino,
            "calmar": calmar,
            "omega": omega,
            "downside_vol": downside_vol,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "tail_ratio": tail_ratio,
            "common_sense_ratio": common_sense_ratio,
            "stability": stability,
            "exposure": exposure,
            "daily_turnover": daily_turnover,
            "best_day": best_day,
            "worst_day": worst_day,
            "best_month": best_month,
            "worst_month": worst_month,
            "best_year": best_year,
            "worst_year": worst_year,
            "longest_dd_duration": longest_dd_duration,
        }
        for k, v in extra_items.items():
            print(f"{k:<20}: {_format_metric_value(k, v)}")

        if len(worst_dd_df) > 0:
            print("\n========== Worst Drawdown Periods ==========")
            print(worst_dd_df.to_string(index=False))

    if show_plots:
        plt.figure(figsize=(12, 5))
        plt.plot(equity.index, equity.values, label="Equity")
        plt.title("Equity Curve")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.fill_between(drawdown.index, drawdown.fillna(0).values, 0)
        plt.title("Drawdown")
        plt.grid(True)
        plt.show()

        if total_trades > 0:
            plt.figure(figsize=(10, 4))
            plt.hist(trade_df["return"].dropna(), bins=20)
            plt.title("Trade Return Distribution")
            plt.grid(True)
            plt.show()

        if len(yearly_ret) > 0:
            plt.figure(figsize=(10, 4))
            plt.bar(yearly_ret.index.year.astype(str), yearly_ret.values)
            plt.title("Yearly Returns")
            plt.grid(True)
            plt.show()

        if len(monthly_ret) > 0:
            plt.figure(figsize=(10, 4))
            plt.hist(monthly_ret.values, bins=20)
            plt.title("Monthly Return Distribution")
            plt.grid(True)
            plt.show()

        if len(rolling_sharpe.dropna()) > 0:
            plt.figure(figsize=(12, 4))
            plt.plot(rolling_sharpe.index, rolling_sharpe.values)
            plt.axhline(0, linestyle="--")
            plt.title(f"Rolling {rolling_window}D Sharpe")
            plt.grid(True)
            plt.show()

        if len(rolling_vol.dropna()) > 0:
            plt.figure(figsize=(12, 4))
            plt.plot(rolling_vol.index, rolling_vol.values)
            plt.title(f"Rolling {rolling_window}D Volatility")
            plt.grid(True)
            plt.show()

        if len(monthly_ret) > 0:
            heat_df = monthly_ret.to_frame("ret")
            heat_df["year"] = heat_df.index.year
            heat_df["month"] = heat_df.index.month
            heat_pivot = heat_df.pivot(index="year", columns="month", values="ret").sort_index()

            plt.figure(figsize=(12, 6))
            plt.imshow(heat_pivot.values, aspect="auto")
            plt.title("Monthly Returns Heatmap")
            plt.yticks(range(len(heat_pivot.index)), heat_pivot.index.astype(str))
            plt.xticks(range(12), range(1, 13))
            plt.colorbar()
            plt.show()

        peak = equity.cummax()
        plt.figure(figsize=(12, 5))
        plt.plot(equity.index, equity.values, label="Equity")
        plt.plot(peak.index, peak.values, linestyle="--", label="Peak")
        plt.title("Equity vs Running Peak")
        plt.grid(True)
        plt.legend()
        plt.show()

    return metrics, trade_df, worst_dd_df