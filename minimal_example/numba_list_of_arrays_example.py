import numpy as np
import time
from numba import njit, types
from numba.typed import List
import numba

array_2d_type = types.float64[:, :]


# 在 njit 函数内部创建并填充 List
@njit
def create_list_of_arrays(shapes_array):
    """
    根据输入的 shapes_array 动态创建一个包含多个2D数组的List。
    """
    # 创建一个类型为 2D 数组的 Numba List
    list_of_arrays = List.empty_list(array_2d_type)

    # 遍历输入的形状数组，动态创建并添加数组
    for value in shapes_array:
        new_array = np.random.rand(value, 6)
        list_of_arrays.append(new_array)

    return list_of_arrays


# --- 更新后的 JIT 工具函数 ---
@njit
def get_item_from_numba_list(data_list, num):
    """
    一个JIT辅助函数，用于从Numba List中取出指定索引的元素。
    """
    assert 0 <= num < len(data_list), "Index out of bounds for Numba List"
    return data_list[num]


# --- 更新后的 JIT 工具函数 ---
@njit
def get_length_from_numba_list(data_list):
    """
    一个JIT辅助函数，用于从Numba List中取出指定索引的元素。
    """
    return len(data_list)


# --- 主测试块 ---
if __name__ == "__main__":
    print("--- 正在进行 Numba 热身... ---")

    # 定义热身用的输入形状数组
    warmup_shapes = np.array([1, 2, 3], dtype=np.int64)

    # 1. 对 create_list_of_arrays 进行热身，并获取其结果
    warmup_created_list = create_list_of_arrays(warmup_shapes)

    # 2. 直接使用上一步得到的数据对 get_item_from_numba_list 进行热身
    get_item_from_numba_list(warmup_created_list, 0)
    get_length_from_numba_list(warmup_created_list)

    print("--- 热身完毕，开始正式测试 ---")

    # 定义输入的形状数组
    input_shapes = np.array([10000, 20000, 50000], dtype=np.int64)

    # 测量 JIT 函数的执行时间 (此时已无编译开销)
    start_time_jit = time.perf_counter()
    result_list = create_list_of_arrays(input_shapes)
    end_time_jit = time.perf_counter()
    print(f"\nnjit函数运行时间 (无编译开销): {end_time_jit - start_time_jit:.6f} 秒")

    start_time_jit = time.perf_counter()

    length = get_length_from_numba_list(result_list)

    print(
        f"返回对象的类型: {type(result_list)}",
        isinstance(result_list, numba.typed.List),
        f"返回子元素对象的类型: {result_list._dtype}",
        result_list._dtype == numba.types.float64[:, :],
    )

    print(f"返回对象的数量: {length}")

    for i in range(length):
        data = get_item_from_numba_list(result_list, i)
        print(f"data: {data[0, 0]} type:{type(data)} shape:{data.shape}")

    end_time_jit = time.perf_counter()
    print(f"\n取值时间 (无编译开销): {end_time_jit - start_time_jit:.6f} 秒")
