from pathlib import Path
import yaml


def load_yaml(file_path: str | Path) -> dict:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"找不到設定檔: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config