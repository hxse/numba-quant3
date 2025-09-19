import polars as pl
from pathlib import Path
import json
import io


def save_data(data, path: Path):
    if isinstance(data, pl.DataFrame):
        if path.suffix == ".csv":
            data.write_csv(path)
        elif path.suffix == ".parquet":
            data.write_parquet(path)
        else:
            raise ValueError(f"Polars DataFrame 不支持的文件后缀: {path.suffix}")
    elif isinstance(data, dict):
        if path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        else:
            raise ValueError(f"字典不支持的文件后缀: {path.suffix}")
    else:
        raise TypeError(f"不支持的数据类型: {type(data)}")


def get_data_buffer(data, path: Path) -> io.BytesIO:
    buffer = io.BytesIO()

    if isinstance(data, pl.DataFrame):
        if path.suffix == ".csv":
            data.write_csv(buffer)
        elif path.suffix == ".parquet":
            data.write_parquet(buffer)
        else:
            raise ValueError(f"Polars DataFrame 不支持的文件后缀: {path.suffix}")
    elif isinstance(data, dict):
        if path.suffix == ".json":
            json_string = json.dumps(data, ensure_ascii=False, indent=4)
            buffer.write(json_string.encode("utf-8"))
        else:
            raise ValueError(f"字典不支持的文件后缀: {path.suffix}")
    else:
        raise TypeError(f"不支持的数据类型: {type(data)}")
    buffer.seek(0)
    return buffer
