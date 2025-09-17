import zipfile
import io
import multiprocessing
from pathlib import Path
from typing import List, Tuple, Any
import zlib


def create_zip_buffer(
    data_list: List[Tuple[Path, io.BytesIO]], compress_level: int = 1
) -> bytes:
    """
    将内存中的文件打包成 ZIP, 并返回 ZIP 压缩包的字节数据
    可以指定压缩级别来调整速度
    如果文件大部分是csv格式, 建议用1
    如果文件大部分是parquet格式, 建议用0
    """
    zip_buffer = io.BytesIO()
    # 注意：zipfile 库本身不支持直接设置压缩级别
    # 但你可以通过手动调用 zlib 来实现

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_STORED) as zipf:
        for path, buffer in data_list:
            buffer.seek(0)
            data = buffer.getvalue()

            # 手动压缩数据
            compressed_data = zlib.compress(data, compress_level)

            # 创建 ZipInfo 对象并设置压缩信息
            info = zipfile.ZipInfo(str(path))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.compress_size = len(compressed_data)
            info.file_size = len(data)

            # 将预先压缩好的数据写入 ZIP
            zipf.writestr(info, compressed_data)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def compress_single_file(
    item: Tuple[Path, io.BytesIO],
) -> Tuple[zipfile.ZipInfo, bytes]:
    """
    在单独的进程中压缩一个文件，并返回压缩后的数据。
    """
    path, buffer = item
    buffer.seek(0)
    data = buffer.getvalue()

    zip_info = zipfile.ZipInfo(str(path))
    zip_info.compress_type = zipfile.ZIP_DEFLATED

    compressed_data = zlib.compress(data, 1)
    zip_info.compress_size = len(compressed_data)
    zip_info.file_size = len(data)

    return zip_info, compressed_data


def create_zip_buffer_parallel(data_list: List[Tuple[Path, io.BytesIO]]) -> bytes:
    """
    使用多进程并行压缩，将内存中的文件打包成 ZIP。
    """
    # 确保在Windows上使用if __name__ == '__main__':
    # 否则在子进程中会再次运行整个脚本
    if __name__ == "__main__":
        multiprocessing.freeze_support()  # 推荐在Windows上使用

    zip_buffer = io.BytesIO()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(compress_single_file, data_list)

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_STORED) as zipf:
        for zip_info, compressed_data in results:
            zipf.writestr(zip_info, compressed_data)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
