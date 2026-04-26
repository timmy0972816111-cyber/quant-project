from src.data.fetchers import fetch_yfinance_data
from src.data.storage import save_csv, load_csv, save_parquet, load_parquet


def main():
    df = fetch_yfinance_data(
        symbol="0050.TW",
        start="2025-01-01",
        end="2025-12-31",
        interval="1d"
    )

    print("成功抓到資料")
    print(df.head())
    print(df.shape)

    save_csv(df, "data/raw/0050_daily.csv", index=False)
    save_parquet(df, "data/processed/0050_daily.parquet", index=False)

    df_csv = load_csv("data/raw/0050_daily.csv")
    df_parquet = load_parquet("data/processed/0050_daily.parquet")

    print("CSV 讀取成功：", df_csv.shape)
    print("Parquet 讀取成功：", df_parquet.shape)


if __name__ == "__main__":
    main()