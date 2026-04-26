from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_heatmap(
    result_df: pd.DataFrame,
    value_col: str = "sharpe_ratio",
    save_path: str = "data/processed/ma_cross_heatmap.png"
) -> None:
    pivot_table = result_df.pivot(
        index="short_window",
        columns="long_window",
        values=value_col
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_table, aspect="auto")
    plt.colorbar(label=value_col)

    plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)

    plt.xlabel("long_window")
    plt.ylabel("short_window")
    plt.title(f"MA Cross Optimization Heatmap ({value_col})")

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()