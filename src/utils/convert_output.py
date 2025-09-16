import pandas as pd
from pathlib import Path
from src.utils.nb_convert_output import jitted_convert_all_dicts
import shutil
import json
import io
import zipfile
import httpx
from typing import Union, List, Tuple
import time

client_session = httpx.Client()


def save_data(data, path: Path):
    if isinstance(data, pd.DataFrame):
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
    if isinstance(data, pd.DataFrame):
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


def create_zip_buffer(data_list: List[Tuple[Path, io.BytesIO]]) -> bytes:
    """
    将内存中的文件打包成 ZIP，并返回 ZIP 压缩包的字节数据。
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path, buffer in data_list:
            buffer.seek(0)
            zipf.writestr(path.name, buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def get_token(token_path):
    with open(token_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data["database-token"]


def upload_to_server(
    client: httpx.Client,
    upload_server: str,
    zip_data: bytes,
    zip_dir_path: Path | None = None,
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
    file_path = zip_dir_path / f"{zip_name if zip_name else 'temp.zip'}"

    # 在函数内部创建 headers
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    start_time = time.perf_counter()

    zip_file_to_upload = io.BytesIO(zip_data)
    files = {
        "file": (
            file_path.as_posix(),
            zip_file_to_upload,
            "application/zip",
        )
    }

    try:
        req_start = time.perf_counter()
        # 将 headers 传递给 post 请求
        response = client.post(f"{upload_server}", files=files, headers=headers)
        response.raise_for_status()
        req_end = time.perf_counter()
        print(f"HTTP 请求部分耗时: {req_end - req_start:.4f} 秒")

        time_elapsed = time.perf_counter() - start_time
        print(f"ZIP 文件已成功上传到服务器，用时 {time_elapsed:.2f} 秒")

    except httpx.HTTPStatusError as e:
        print(f"上传文件到服务器失败: HTTP 状态码错误 - {e}")
    except httpx.RequestError as e:
        print(f"上传文件到服务器失败: 请求错误 - {e}")
    except Exception as e:
        print(f"上传文件到服务器失败: 未知错误 - {e}")


def process_and_archive(
    data_list: List[Tuple[str, pd.DataFrame | dict]],
    save_local_dir: str | Path | None = None,
    save_zip_dir: str | Path | None = None,
    upload_server: str | None = None,
    token_path: str | None = None,
    zip_name: str = "strategy.zip",
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
            # print(f"原始文件已保存到本地: {file_path}")

    # 2. 如果需要 ZIP 或上传，则生成 ZIP 数据
    if zip_dir_path or upload_server:
        data_buffers = []
        for name, data in data_list:
            # 确保 get_data_buffer 得到 Path 对象
            buffer = get_data_buffer(data, Path(name))
            data_buffers.append((Path(name), buffer))

        zip_data = create_zip_buffer(data_buffers)
        del data_buffers  # 及时释放内存

        # 3. 可选操作：保存 ZIP 文件到本地
        if zip_dir_path:
            save_zip_path = zip_dir_path / zip_name
            save_zip_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_zip_path, "wb") as f:
                f.write(zip_data)
            # print(f"ZIP 文件已保存到本地: {save_zip_path}")

        # 4. 可选操作：上传 ZIP 文件到服务器

        if upload_server:
            assert token_path, "需要token"
            token = get_token(token_path)
            upload_to_server(
                client=client_session,
                upload_server=upload_server,
                zip_data=zip_data,
                zip_dir_path=zip_dir_path,
                zip_name=zip_name,
                token=token,
            )


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


def convert_output(
    params: tuple,
    result: tuple,
    num: int = 0,
    data_suffix: str = ".csv",
    params_suffix: str = ".json",
    save_local_dir: str | Path | None = None,
    save_zip_dir: str | Path | None = None,
    upload_server: str | None = None,
    token_path: str | None = None,
    zip_name: str = "strategy.zip",
):
    """
    转换数据并处理输出。
    """
    save_local_dir = Path(save_local_dir) if save_local_dir else None
    save_zip_dir = Path(save_zip_dir) if save_zip_dir else None

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
            df = pd.DataFrame(np_item, columns=keys)
            final_result[name] = df
            data_list.append((f"{name}{data_suffix}", df))
        else:
            raise RuntimeError(f"检测到未预期维度数 {len(np_item.shape)}")

    # 调用通用的处理函数
    process_and_archive(
        data_list,
        save_local_dir=save_local_dir,
        save_zip_dir=save_zip_dir,
        upload_server=upload_server,
        token_path=token_path,
        zip_name=zip_name,
    )

    return final_result
