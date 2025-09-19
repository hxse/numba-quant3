from pathlib import Path


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
