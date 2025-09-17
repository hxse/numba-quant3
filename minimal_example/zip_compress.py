import zipfile
import io
import multiprocessing
from pathlib import Path
import zlib
import time
from typing import List, Tuple
import pandas as pd
import numpy as np


# 你的单进程 ZIP 压缩函数
def create_zip_buffer(data_list: List[Tuple[Path, io.BytesIO]]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path, buffer in data_list:
            buffer.seek(0)
            zipf.writestr(path.name, buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# 单个文件的压缩任务函数 (用于多进程)
def compress_single_file(
    item: Tuple[Path, io.BytesIO],
) -> Tuple[zipfile.ZipInfo, bytes]:
    path, buffer = item
    buffer.seek(0)
    data = buffer.getvalue()
    zip_info = zipfile.ZipInfo(str(path))
    zip_info.compress_type = zipfile.ZIP_DEFLATED
    compressed_data = zlib.compress(data, zlib.Z_DEFAULT_COMPRESSION)
    zip_info.compress_size = len(compressed_data)
    zip_info.file_size = len(data)
    return zip_info, compressed_data


# 你的多进程 ZIP 压缩函数
def create_zip_buffer_parallel(data_list: List[Tuple[Path, io.BytesIO]]) -> bytes:
    zip_buffer = io.BytesIO()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(compress_single_file, data_list)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_STORED) as zipf:
        for zip_info, compressed_data in results:
            zipf.writestr(zip_info, compressed_data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# --- 主测试代码 ---
def run_benchmark(num_rows: int, description: str):
    """运行一次基准测试，并打印结果。"""
    print(f"\n--- 正在测试: {description} ---")

    # 1. 生成模拟数据
    print("生成测试数据...")
    data_list = []
    # 创建 10 个 DataFrame
    for i in range(10):
        df = pd.DataFrame(
            {
                "col1": np.random.rand(num_rows),
                "col2": np.random.randint(0, 100, num_rows),
            }
        )
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        data_list.append((Path(f"file_{i}.csv"), buffer))

    # 2. 测试单进程 ZIP 压缩
    print("\n--- 单进程 ZIP 压缩 ---")
    start_time_single = time.perf_counter()
    zip_data_single = create_zip_buffer(data_list)
    end_time_single = time.perf_counter()
    time_single = end_time_single - start_time_single
    print(f"用时: {time_single:.4f} 秒")
    print(f"生成的 ZIP 文件大小: {len(zip_data_single) / 1024 / 1024:.2f} MB")

    # 3. 测试多进程 ZIP 压缩
    print("\n--- 多进程 ZIP 压缩 ---")
    start_time_parallel = time.perf_counter()
    try:
        zip_data_parallel = create_zip_buffer_parallel(data_list)
        end_time_parallel = time.perf_counter()
        time_parallel = end_time_parallel - start_time_parallel
        print(f"用时: {time_parallel:.4f} 秒")
        print(f"生成的 ZIP 文件大小: {len(zip_data_parallel) / 1024 / 1024:.2f} MB")

        if time_parallel < time_single:
            print(
                f"结论：多进程比单进程快了 {(time_single - time_parallel) / time_single * 100:.2f}%"
            )
        else:
            print(
                f"结论：单进程比多进程快了 {(time_parallel - time_single) / time_single * 100:.2f}%"
            )

    except Exception as e:
        print(f"多进程测试失败，出现错误: {e}")
        print("这通常是由于多进程序列化问题或外部依赖导致的。")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 测试大数据量，每个文件 50 万行
    num_rows = 10_000 * 50
    run_benchmark(num_rows=num_rows, description=f"大数据量测试 {num_rows}")

    # 测试小数据量，每个文件 1 万行
    num_rows = 10_000
    run_benchmark(num_rows=num_rows, description=f"小数据量测试 {num_rows}")
