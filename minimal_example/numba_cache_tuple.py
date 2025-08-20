import numpy as np
from numba import njit, types, from_dtype
from numba.typed import Dict, List
import os
import time
import shutil


# 定义一个 Numba 函数，接受一个元组参数
@njit(cache=True)
def process_data(data_dict, keys_tuple):
    # 添加一个检查，如果元组为空，直接返回
    if len(keys_tuple) == 0:
        return 0.0

    total = 0.0
    for k in keys_tuple:
        total += data_dict.get(k, 0.0)
    return total


def setup_data():
    """创建一个 Numba Typed Dict 作为测试数据"""
    data = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    data["one"] = 1.1
    data["two"] = 2.2
    data["three"] = 3.3
    return data


def main():
    # 1. 确保缓存目录是空的，这样结果更清晰
    cache_dir = "minimal_example/__pycache__"
    # if os.path.exists(cache_dir):
    #     shutil.rmtree(cache_dir)
    #     print(f"已清理旧的缓存目录: {cache_dir}")
    # os.makedirs(cache_dir, exist_ok=True)

    data_dict = setup_data()

    # 2. 定义不同类型的元组
    empty_tuple = ()  # 空元组
    single_item_tuple = ("one",)  # 单个元素的元组
    two_items_tuple = ("one", "two")  # 两个元素的元组

    # 3. 第一次调用，会触发编译和缓存
    print("\n--- 第一次调用 (会触发编译) ---")
    start_time = time.time()
    result_a = process_data(data_dict, empty_tuple)
    duration_a = time.time() - start_time
    print(f"空元组调用结果: {result_a}")
    print(f"空元组调用耗时: {duration_a:.4f} 秒")

    start_time = time.time()
    result_b = process_data(data_dict, single_item_tuple)
    duration_b = time.time() - start_time
    print(f"单元素元组调用结果: {result_b}")
    print(f"单元素元组调用耗时: {duration_b:.4f} 秒")

    start_time = time.time()
    result_c = process_data(data_dict, two_items_tuple)
    duration_c = time.time() - start_time
    print(f"双元素元组调用结果: {result_c}")
    print(f"双元素元组调用耗时: {duration_c:.4f} 秒")

    # 4. 打印缓存文件数量
    num_files = len([f for f in os.listdir(cache_dir) if f.endswith(".nbc")])
    print(f"\n--- 第一次运行后，缓存目录中的 .nbc 文件数量: {num_files} ---")

    print("理论上你应该看到 3 个 .nbc 文件，对应 3 次不同的编译。")

    # 5. 第二次调用，应该直接使用缓存，耗时会非常短
    print("\n--- 第二次调用 (直接使用缓存) ---")
    start_time = time.time()
    process_data(data_dict, empty_tuple)
    duration_a2 = time.time() - start_time
    print(f"空元组第二次调用耗时: {duration_a2:.4f} 秒")

    start_time = time.time()
    process_data(data_dict, single_item_tuple)
    duration_b2 = time.time() - start_time
    print(f"单元素元组第二次调用耗时: {duration_b2:.4f} 秒")

    start_time = time.time()
    process_data(data_dict, two_items_tuple)
    duration_c2 = time.time() - start_time
    print(f"双元素元组第二次调用耗时: {duration_c2:.4f} 秒")


if __name__ == "__main__":
    main()
