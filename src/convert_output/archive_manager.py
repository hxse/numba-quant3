import polars as pl
from pathlib import Path
import time
from typing import List, Tuple
from src.convert_output.data_io import save_data, get_data_buffer
from src.convert_output.server_upload import upload_to_server
from src.convert_output.directory_utils import clean_directory
from src.utils.parallel_compress import create_zip_buffer

import httpx  # 导入 httpx


client_session = httpx.Client()


def archive_data(
    data_list: List[Tuple[str, pl.DataFrame | dict]],
    save_local_dir: str | Path | None = None,
    save_zip_dir: str | Path | None = None,
    upload_server: str | None = None,
    server_dir: str | None = None,
    zip_name: str = "strategy.zip",
    compress_level: int = 1,
    username: str | None = None,
    password: str | None = None,
    client_session: httpx.Client = client_session,
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
            upload_to_server(
                client=client_session,  # 使用传入的 client_session
                upload_server=upload_server,
                zip_data=zip_data,
                server_dir=server_dir,
                zip_name=zip_name,
                username=username,
                password=password,
            )
