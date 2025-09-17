import pandas as pd
import polars as pl
from pathlib import Path
from src.utils.nb_convert_output import jitted_convert_all_dicts
import shutil
import json
import io
import zipfile
import httpx
from typing import Union, List, Tuple
import time
from src.utils.parallel_compress import create_zip_buffer_parallel, create_zip_buffer

client_session = httpx.Client()


def save_data(data, path: Path):
    if isinstance(data, pl.DataFrame):
        if path.suffix == ".csv":
            data.write_csv(path)
        elif path.suffix == ".parquet":
            data.write_parquet(path)
        else:
            raise ValueError(f"Polars DataFrame 不支持的文件后缀: {path.suffix}")
    elif isinstance(data, pd.DataFrame):
        if path.suffix == ".csv":
            data.to_csv(path, index=False, encoding="utf-8")
        elif path.suffix == ".parquet":
            data.to_parquet(path, index=False)
        else:
            raise ValueError(f"DataFrame 不支持的文件后缀: {path.suffix}")
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
    elif isinstance(data, pd.DataFrame):
        if path.suffix == ".csv":
            data.to_csv(buffer, index=False, encoding="utf-8")
        elif path.suffix == ".parquet":
            data.to_parquet(buffer, index=False)
        else:
            raise ValueError(f"DataFrame 不支持的文件后缀: {path.suffix}")
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


def get_token(token_path):
    with open(token_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data["database-token"]


def upload_to_server(
    client: httpx.Client,
    upload_server: str,
    zip_data: bytes,
    server_dir: Path | None = None,
    zip_name: str = "strategy.zip",
    token: str | None = None,
):
    """
    将 ZIP 文件的字节数据上传到指定的服务器。

    参数:
    client (httpx.Client): 一个已初始化的 httpx 客户端实例。
    upload_server (str): 服务器的 URL。
    zip_data (bytes): ZIP 文件的字节数据。
    zip_dir_path (Path | None): ZIP 文件所在的目录路径，用于生成默认文件名。
    zip_name (str): ZIP 文件的名称，默认为 "strategy.zip"。
    token (Optional[str]): 用于 HTTPBearer 认证的令牌。
    """
    start_time = time.perf_counter()

    # 在函数内部创建 headers
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    file_path = (
        Path(server_dir) if server_dir else Path("./")
    ) / f"{zip_name if zip_name else 'temp.zip'}"

    zip_file_to_upload = io.BytesIO(zip_data)
    files = {
        "file": (
            file_path.as_posix(),
            zip_file_to_upload,
            "application/zip",
        )
    }

    try:
        # 将 headers 传递给 post 请求
        response = client.post(f"{upload_server}", files=files, headers=headers)
        response.raise_for_status()

        time_elapsed = time.perf_counter() - start_time
        print(f"ZIP 文件已成功上传到服务器，用时 {time_elapsed:.2f} 秒")

    except httpx.HTTPStatusError as e:
        print(f"上传文件到服务器失败: HTTP 状态码错误 - {e}")
    except httpx.RequestError as e:
        print(f"上传文件到服务器失败: 请求错误 - {e}")
    except Exception as e:
        print(f"上传文件到服务器失败: 未知错误 - {e}")


def clean_directory(dir_path: Path, suffixes: list[str]):
    """
    清理指定目录中特定后缀的文件。

    参数:
    dir_path (Path): 要清理的目录路径。
    suffixes (list[str]): 要删除的文件后缀列表，例如 ['.csv', '.json']。
    """
    if not dir_path.is_dir():
        # print(f"警告：目录不存在，跳过清理：{dir_path}")
        return

    # print(f"开始清理目录：{dir_path}")
    for file in dir_path.iterdir():
        if file.suffix in suffixes:
            try:
                file.unlink()
                # print(f"已删除文件：{file}")
            except OSError as e:
                print(f"删除文件失败 {file}: {e}")
    # print("目录清理完成。")


def archive_data(
    data_list: List[Tuple[str, pd.DataFrame | dict]],
    save_local_dir: str | Path | None = None,
    save_zip_dir: str | Path | None = None,
    upload_server: str | None = None,
    server_dir: str | None = None,
    token_path: str | None = None,
    zip_name: str = "strategy.zip",
    compress_level: int = 1,
) -> None:
    """
    处理数据列表，可选地保存原始文件，然后打包成 ZIP，
    并可选地保存 ZIP 到本地或上传到服务器。
    """
    local_path = Path(save_local_dir) if save_local_dir else None
    zip_dir_path = Path(save_zip_dir) if save_zip_dir else None

    suffix_list = [".csv", ".parquet", ".json", ".zip"]
    if local_path:
        clean_directory(local_path, suffix_list)

    if zip_dir_path:
        clean_directory(zip_dir_path, suffix_list)

    # 1. 可选操作：保存原始文件到本地
    if local_path:
        local_path.mkdir(parents=True, exist_ok=True)
        for name, data in data_list:
            file_path = local_path / name
            save_data(data, file_path)

    # 2. 如果需要 ZIP 或上传，则生成 ZIP 数据
    if zip_dir_path or upload_server:
        zip_start_time = time.perf_counter()
        data_buffers = []
        for name, data in data_list:
            # 确保 get_data_buffer 得到 Path 对象
            buffer = get_data_buffer(data, Path(name))
            data_buffers.append((Path(name), buffer))
        # print(f"buffers生成用时, {time.perf_counter() - zip_start_time}")

        # zip_data = create_zip_buffer_parallel(data_buffers) #实测会慢, 因为数据太少了
        zip_data = create_zip_buffer(data_buffers, compress_level=compress_level)
        del data_buffers  # 及时释放内存

        # 3. 可选操作：保存 ZIP 文件到本地
        if zip_dir_path:
            save_zip_path = zip_dir_path / zip_name
            save_zip_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_zip_path, "wb") as f:
                f.write(zip_data)
        # print(f"zip生成用时, {time.perf_counter() - zip_start_time}")

        # 4. 可选操作：上传 ZIP 文件到服务器
        if upload_server:
            assert token_path, "需要token"
            token = get_token(token_path)
            upload_to_server(
                client=client_session,
                upload_server=upload_server,
                zip_data=zip_data,
                server_dir=server_dir,
                zip_name=zip_name,
                token=token,
            )


def convert_output(
    params: tuple,
    result: tuple,
    num: int = 0,
    data_suffix: str = ".csv",
    params_suffix: str = ".json",
):
    """
    转换数据并处理输出。
    upload_server用127.0.0.1, 别用localhost, 会慢
    """

    result_converted = jitted_convert_all_dicts(params, result, num)
    final_result = {}
    data_list = []

    for name, keys_item, dict_item, np_item in result_converted:
        keys = tuple(keys_item)
        if len(np_item.shape) == 1:
            _dict = {k: float(v) for k, v in zip(keys, np_item)}
            final_result[name] = _dict
            data_list.append((f"{name}{params_suffix}", _dict))
        elif len(np_item.shape) == 2:
            # df = pd.DataFrame(np_item, columns=keys)
            df = pl.from_numpy(np_item, schema=keys)
            final_result[name] = df
            data_list.append((f"{name}{data_suffix}", df))
        else:
            raise RuntimeError(f"检测到未预期维度数 {len(np_item.shape)}")

    return final_result, data_list
