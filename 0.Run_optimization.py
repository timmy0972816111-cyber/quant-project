from src.optimization.grid_search import run_ma_cross_grid_search
from src.optimization.visualization import plot_heatmap


def main():
    result_df, best_params = run_ma_cross_grid_search("config/optimization.yaml")

    print("\n=== 最佳化結果前 10 名 ===")
    print(result_df.head(10))

    print("\n=== 最佳參數 ===")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    plot_heatmap(
        result_df,
        value_col="sharpe_ratio",
        save_path="data/processed/ma_cross_sharpe_heatmap.png"
    )

    plot_heatmap(
        result_df,
        value_col="total_return",
        save_path="data/processed/ma_cross_return_heatmap.png"
    )

    print("\n熱圖已輸出到 data/processed/")
    print("1. ma_cross_sharpe_heatmap.png")
    print("2. ma_cross_return_heatmap.png")


if __name__ == "__main__":
    main()