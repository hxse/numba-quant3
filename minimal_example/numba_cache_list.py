import numpy as np
from numba import njit, types
from numba.typed import Dict, List
import os
import time
import shutil


@njit(cache=True)
def process_data_with_list(data_dict, keys_list):
    total = 0.0
    for k in keys_list:
        total += data_dict.get(k, 0.0)
    return total


def setup_data():
    data = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    data["one"] = 1.1
    data["two"] = 2.2
    data["three"] = 3.3
    return data


def main():
    cache_dir = "__pycache__"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    data_dict = setup_data()

    # 创建不同长度的列表
    empty_list = List.empty_list(types.unicode_type)
    single_item_list = List.empty_list(types.unicode_type)
    single_item_list.append("one")
    two_items_list = List.empty_list(types.unicode_type)
    two_items_list.append("one")
    two_items_list.append("two")

    print("\n--- 第一次调用 (会触发编译) ---")

    start_time = time.time()
    result_a = process_data_with_list(data_dict, empty_list)
    duration_a = time.time() - start_time
    print(f"空列表调用耗时: {duration_a:.4f} 秒")

    start_time = time.time()
    result_b = process_data_with_list(data_dict, single_item_list)
    duration_b = time.time() - start_time
    print(f"单元素列表调用耗时: {duration_b:.4f} 秒")

    start_time = time.time()
    result_c = process_data_with_list(data_dict, two_items_list)
    duration_c = time.time() - start_time
    print(f"双元素列表调用耗时: {duration_c:.4f} 秒")

    num_files = len([f for f in os.listdir(cache_dir) if f.endswith(".nbc")])
    print(f"\n--- 第一次运行后，缓存目录中的 .nbc 文件数量: {num_files} ---")
    print("理论上你应该只看到 1 个 .nbc 文件。")


if __name__ == "__main__":
    main()
