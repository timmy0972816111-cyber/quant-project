from pathlib import Path
import pandas as pd


def ensure_folder(folder_path: str | Path) -> Path:
    """
    確保資料夾存在，不存在就建立
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_csv(df: pd.DataFrame, file_path: str | Path, index: bool = True) -> None:
    """
    將 DataFrame 存成 CSV
    """
    file_path = Path(file_path)
    ensure_folder(file_path.parent)
    df.to_csv(file_path, index=index, encoding="utf-8-sig")


def load_csv(file_path: str | Path, **kwargs) -> pd.DataFrame:
    """
    讀取 CSV 成 DataFrame
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到檔案: {file_path}")
    return pd.read_csv(file_path, encoding="utf-8-sig", **kwargs)


def save_parquet(df: pd.DataFrame, file_path: str | Path, index: bool = True) -> None:
    """
    將 DataFrame 存成 parquet
    """
    file_path = Path(file_path)
    ensure_folder(file_path.parent)
    df.to_parquet(file_path, index=index)


def load_parquet(file_path: str | Path, **kwargs) -> pd.DataFrame:
    """
    讀取 parquet 成 DataFrame
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到檔案: {file_path}")
    return pd.read_parquet(file_path, **kwargs)