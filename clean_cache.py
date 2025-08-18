import shutil
from pathlib import Path


def clean_pycache_pathlib(root_dir="."):
    """
    使用 pathlib 递归清理指定目录下所有的 __pycache__ 文件夹。

    Args:
        root_dir (str): 开始遍历的根目录。
    """
    root_path = Path(root_dir)

    # 检查当前目录是否为项目根目录，通过是否存在 pyproject.toml 文件来判断
    pyproject_toml_path = root_path / "pyproject.toml"
    assert pyproject_toml_path.is_file(), (
        f"未在 '{root_path}' 目录中找到 'pyproject.toml' 文件。 "
        "请确保你在项目根目录中运行此脚本。"
    )

    print("开始清理 __pycache__ 文件夹...")
    print("-" * 30)

    # 使用 rglob 遍历所有 __pycache__ 文件夹，无论它们嵌套多深
    # exclude_dirs 列表用于手动检查和跳过
    exclude_dirs = {".git", ".venv"}

    for path in root_path.rglob("__pycache__"):
        # 检查当前路径的父级路径，判断是否在需要跳过的文件夹中
        is_in_excluded_dir = False
        for part in path.parts:
            if part in exclude_dirs:
                is_in_excluded_dir = True
                break

        if is_in_excluded_dir:
            # print(f"跳过（在排除目录中）: {path}")
            continue

        print(f"正在移除: {path}")
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"错误: {path} 无法删除。{e}")

    print("-" * 30)
    print("清理完成！")


if __name__ == "__main__":
    clean_pycache_pathlib()
