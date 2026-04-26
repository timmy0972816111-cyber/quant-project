import numpy as np
import pandas as pd


SCORED_METRICS = {
    "Sharpe": {"weight": 1.5},
    "Sortino": {"weight": 1.2},
    "Calmar": {"weight": 1.5},
    "MDD": {"weight": 1.5},
    "Profit Factor": {"weight": 1.2},
    "Expectancy": {"weight": 1.2},
    "Stability": {"weight": 1.0},
    "CAGR": {"weight": 0.8},
    "Payoff Ratio": {"weight": 0.8},
}


# 這些指標適合用百分比顯示
PERCENT_DISPLAY_METRICS = {
    "Total Return",
    "CAGR",
    "Volatility",
    "MDD",
    "Win Rate",
    "Expectancy",
    "VaR 95%",
    "CVaR 95%",
    "Exposure",
    "Daily Turnover",
}


# 這些指標適合用整數顯示
INTEGER_DISPLAY_METRICS = {
    "Longest DD Days",
}


def _format_display_value(metric_name, value):
    if value is None:
        return "NaN"

    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return "NaN"

    if metric_name in PERCENT_DISPLAY_METRICS:
        return f"{value:.2%}"

    if metric_name in INTEGER_DISPLAY_METRICS:
        return f"{int(value)}"

    if isinstance(value, (int, float, np.floating)):
        return f"{value:.3f}"

    return str(value)


def _grade_higher_better(value, thresholds):
    if pd.isna(value):
        return None, "無資料"
    if value < thresholds[0]:
        return 1, "差"
    elif value < thresholds[1]:
        return 2, "普通"
    elif value < thresholds[2]:
        return 3, "良好"
    else:
        return 4, "優秀"


def _grade_lower_better_abs(value, thresholds):
    if pd.isna(value):
        return None, "無資料"
    v = abs(value)
    if v > thresholds[2]:
        return 1, "差"
    elif v > thresholds[1]:
        return 2, "普通"
    elif v > thresholds[0]:
        return 3, "良好"
    else:
        return 4, "優秀"

def _comment(metric, value):
    if pd.isna(value):
        return "無資料，暫時無法判讀。"

    # ===== 有分數的核心指標：保留評價型 comment =====
    if metric == "Sharpe":
        if value < 0:
            return "Sharpe 為負，代表承擔風險卻沒有得到合理報酬。"
        elif value < 0.5:
            return "Sharpe 偏弱，風險調整後績效有限。"
        elif value < 1.0:
            return "Sharpe 不錯，策略具備基本實用性。"
        else:
            return "Sharpe 良好，策略具備不錯的風險調整報酬。"

    elif metric == "Sortino":
        if value < 0:
            return "Sortino 為負，代表下行風險下的報酬補償不足。"
        elif value < 0.7:
            return "Sortino 普通，對壞波動的補償不算強。"
        elif value < 1.2:
            return "Sortino 良好，對 downside risk 的處理不錯。"
        else:
            return "Sortino 很強，對下行風險的控制品質佳。"

    elif metric == "Calmar":
        if value < 0.2:
            return "Calmar 偏弱，報酬相對最大回撤不夠漂亮。"
        elif value < 0.5:
            return "Calmar 普通，報酬與回撤大致平衡。"
        elif value < 1.0:
            return "Calmar 良好，報酬相對回撤已有吸引力。"
        else:
            return "Calmar 很強，代表策略用相對合理回撤換到不錯報酬。"

    elif metric == "MDD":
        v = abs(value)
        if v > 0.30:
            return "最大回撤偏深，持有壓力相當明顯。"
        elif v > 0.20:
            return "最大回撤偏大，需搭配較高報酬才合理。"
        elif v > 0.10:
            return "最大回撤可接受，屬一般策略常見範圍。"
        else:
            return "最大回撤小，資金曲線相對穩定。"

    elif metric == "Profit Factor":
        if value < 1.0:
            return "總獲利小於總虧損，策略整體不具優勢。"
        elif value < 1.3:
            return "策略有正期望，但優勢不算明顯。"
        elif value < 1.8:
            return "獲利品質良好，總獲利明顯高於總虧損。"
        else:
            return "獲利品質很強，交易結構相對健康。"

    elif metric == "Expectancy":
        if value < 0:
            return "每筆交易期望值為負，長期執行不利。"
        elif value < 0.001:
            return "每筆交易 edge 很薄，成本與滑價容易侵蝕優勢。"
        elif value < 0.005:
            return "每筆交易期望值不錯，具備可用 edge。"
        else:
            return "每筆交易期望值很強，出手機會品質佳。"

    elif metric == "Stability":
        if value < 0.3:
            return "資金曲線不穩，報酬路徑較扭曲。"
        elif value < 0.6:
            return "資金曲線穩定度普通。"
        elif value < 0.8:
            return "資金曲線穩定度良好。"
        else:
            return "資金曲線非常平順，穩定性高。"

    elif metric == "CAGR":
        if value < 0:
            return "年化報酬為負，長期複利效果不成立。"
        elif value < 0.08:
            return "年化報酬偏低，但仍需搭配風險指標一起判讀。"
        elif value < 0.18:
            return "年化報酬不錯，具備一定策略價值。"
        else:
            return "年化報酬良好，若風險控制得當值得進一步研究。"

    elif metric == "Payoff Ratio":
        if value < 0.8:
            return "平均獲利小於平均虧損，賺賠結構偏弱。"
        elif value < 1.2:
            return "賺賠比普通，沒有明顯優勢。"
        elif value < 1.8:
            return "平均獲利大於平均虧損，結構不錯。"
        else:
            return "賺賠結構很強，屬於明顯賺大於賠的型態。"

    # ===== 描述型指標：改成定義 + 簡單解釋 =====
    elif metric == "Total Return":
        return "用來看策略從頭到尾總共賺了多少或虧了多少，但會受回測期間長短影響。"

    elif metric == "Volatility":
        return "數值越大代表績效起伏越大，但不能單獨判斷好壞，需搭配 Sharpe 或 Calmar 一起看。"

    elif metric == "Win Rate":
        return "反映策略命中率，但高勝率不一定代表策略好，仍需搭配賺賠比與 Profit Factor 一起判讀。"

    elif metric == "Skewness":
        return "正偏通常代表偶爾有較大的獲利，負偏則表示較容易出現大虧損。"

    elif metric == "Kurtosis":
        return "數值越高通常代表肥尾越明顯，也就是極端波動發生的機率較高。"

    elif metric == "Tail Ratio":
        return "大於 1 通常表示正向尾部比負向尾部更強，代表大賺的潛力相對較高。"

    elif metric == "VaR 95%":
        return "可理解為一般情況下最差 5% 時的損失邊界，但無法描述更極端情況的平均損失。"

    elif metric == "CVaR 95%":
        return "比 VaR 更能反映真正壞情況下的尾部風險。"

    elif metric == "Exposure":
        return "數值越高代表策略越常待在市場中，偏向高參與型；數值越低則代表策略較挑進場時機。"

    elif metric == "Daily Turnover":
        return "數值越高通常代表交易越頻繁，也意味著成本與滑價影響可能更明顯。"

    elif metric == "Longest DD Days":
        return "這個指標反映資金修復速度與持有耐心壓力，時間越長通常越考驗紀律。"

    return "此指標已有數值，但目前僅作描述，不納入總評分。"

def diagnose_performance(metrics: dict, verbose: bool = True):
    diagnosis_rows = []

    def add_row(metric_name, value, score=None, grade=None, weight=None, scored=False):
        diagnosis_rows.append({
            "Metric": metric_name,
            "Value": value,
            "Grade": grade if grade is not None else "描述",
            "Score": score,
            "Weight": weight,
            "Scored": scored,
            "Comment": _comment(metric_name, value),
        })

    # --- 有分數的核心指標 ---
    score, grade = _grade_higher_better(metrics.get("sharpe"), [0.0, 0.5, 1.0])
    add_row("Sharpe", metrics.get("sharpe"), score, grade, SCORED_METRICS["Sharpe"]["weight"], True)

    score, grade = _grade_higher_better(metrics.get("sortino"), [0.0, 0.7, 1.2])
    add_row("Sortino", metrics.get("sortino"), score, grade, SCORED_METRICS["Sortino"]["weight"], True)

    score, grade = _grade_higher_better(metrics.get("calmar"), [0.2, 0.5, 1.0])
    add_row("Calmar", metrics.get("calmar"), score, grade, SCORED_METRICS["Calmar"]["weight"], True)

    score, grade = _grade_lower_better_abs(metrics.get("mdd"), [0.10, 0.20, 0.30])
    add_row("MDD", metrics.get("mdd"), score, grade, SCORED_METRICS["MDD"]["weight"], True)

    score, grade = _grade_higher_better(metrics.get("profit_factor"), [1.0, 1.3, 1.8])
    add_row("Profit Factor", metrics.get("profit_factor"), score, grade, SCORED_METRICS["Profit Factor"]["weight"], True)

    score, grade = _grade_higher_better(metrics.get("expectancy"), [0.0, 0.001, 0.005])
    add_row("Expectancy", metrics.get("expectancy"), score, grade, SCORED_METRICS["Expectancy"]["weight"], True)

    score, grade = _grade_higher_better(metrics.get("stability"), [0.3, 0.6, 0.8])
    add_row("Stability", metrics.get("stability"), score, grade, SCORED_METRICS["Stability"]["weight"], True)

    score, grade = _grade_higher_better(metrics.get("cagr"), [0.0, 0.08, 0.18])
    add_row("CAGR", metrics.get("cagr"), score, grade, SCORED_METRICS["CAGR"]["weight"], True)

    score, grade = _grade_higher_better(metrics.get("payoff_ratio"), [0.8, 1.2, 1.8])
    add_row("Payoff Ratio", metrics.get("payoff_ratio"), score, grade, SCORED_METRICS["Payoff Ratio"]["weight"], True)

    # --- 只評論，不打分 ---
    descriptive_metrics = {
        "Total Return": metrics.get("total_return"),
        "Volatility": metrics.get("vol"),
        "Win Rate": metrics.get("win_rate"),
        "Skewness": metrics.get("skewness"),
        "Kurtosis": metrics.get("kurtosis"),
        "Tail Ratio": metrics.get("tail_ratio"),
        "VaR 95%": metrics.get("var_95"),
        "CVaR 95%": metrics.get("cvar_95"),
        "Exposure": metrics.get("exposure"),
        "Daily Turnover": metrics.get("daily_turnover"),
        "Longest DD Days": metrics.get("longest_dd_duration"),
    }

    for metric_name, value in descriptive_metrics.items():
        add_row(metric_name, value, score=None, grade="描述", weight=None, scored=False)

    diagnosis_df = pd.DataFrame(diagnosis_rows)

    scored_df = diagnosis_df[diagnosis_df["Scored"] == True].copy()
    scored_df = scored_df.dropna(subset=["Score", "Weight"])

    if len(scored_df) == 0:
        avg_score = np.nan
        overall_comment = "目前沒有足夠可評分指標，無法產出總評。"
    else:
        weighted_score = (scored_df["Score"] * scored_df["Weight"]).sum() / scored_df["Weight"].sum()
        avg_score = weighted_score

        if avg_score < 1.8:
            overall_comment = "整體評估偏弱，策略尚未具備穩定實戰價值。"
        elif avg_score < 2.5:
            overall_comment = "整體評估普通，具備部分優點，但仍有明顯缺陷需修正。"
        elif avg_score < 3.2:
            overall_comment = "整體評估良好，策略已有一定可用性，建議進一步做樣本外與穩健性測試。"
        else:
            overall_comment = "整體評估優秀，若樣本外、成本與實盤模擬也通過，具備高度研究價值。"

    if verbose:
        print("\n========== Strategy Diagnosis ==========")
        for _, row in diagnosis_df.iterrows():
            val_text = _format_display_value(row["Metric"], row["Value"])
            print(f"[{row['Metric']}] {val_text} | 評價：{row['Grade']} | {row['Comment']}")

        print("\n========== Overall Diagnosis ==========")
        print(f"加權平均分數：{avg_score:.2f}" if not pd.isna(avg_score) else "加權平均分數：無法計算")
        print(f"總評：{overall_comment}")

    return diagnosis_df, avg_score, overall_comment