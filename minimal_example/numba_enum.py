import numpy as np
from numba import njit
from enum import IntEnum, unique
import time


# --- 1. 定义 IntEnum 类 ---
# 保留 @unique 和 IntEnum
@unique
class PositionStatus(IntEnum):
    """定义回测中的仓位状态。"""

    NO_POSITION = 0
    LONG = 1
    SHORT = -1


# --- 2. 编写 Numba 兼容的辅助函数 ---
# 将静态方法重写为独立的 njit 函数
@njit
def is_long_position(status_int):
    """检查仓位状态是否为多头。"""
    return status_int == PositionStatus.LONG


@njit
def is_short_position(status_int):
    """检查仓位状态是否为空头。"""
    return status_int == PositionStatus.SHORT


# --- 3. 编写使用辅助函数的 njit 主函数 ---
@njit
def check_position_logic(positions_array):
    """
    一个 njit 函数，用于检查仓位数组中的状态。
    这个函数会调用 Numba 兼容的辅助函数。
    """
    long_count = 0
    short_count = 0
    no_position_count = 0

    for status_int in positions_array:
        if is_long_position(status_int):
            long_count += 1
        elif is_short_position(status_int):
            short_count += 1
        elif status_int == PositionStatus.NO_POSITION:
            no_position_count += 1

    return long_count, short_count, no_position_count


# --- 4. 运行测试 ---
if __name__ == "__main__":
    # 创建一个包含仓位状态的 NumPy 数组
    test_array = np.array([1, -1, 0, 1, -1, 0, 1, 1, 0, -1], dtype=np.int64)

    print("--- 首次运行 (带编译) ---")
    start_time = time.perf_counter()
    long, short, none = check_position_logic(test_array)
    end_time = time.perf_counter()

    print(f"多头仓位数: {long}")
    print(f"空头仓位数: {short}")
    print(f"无仓位数: {none}")
    print(f"首次运行时间: {end_time - start_time:.6f} 秒")
    print("-" * 30)

    print("--- 第二次运行 (无编译) ---")
    start_time = time.perf_counter()
    long, short, none = check_position_logic(test_array)
    end_time = time.perf_counter()

    print(f"多头仓位数: {long}")
    print(f"空头仓位数: {short}")
    print(f"无仓位数: {none}")
    print(f"第二次运行时间: {end_time - start_time:.6f} 秒")
    print("-" * 30)
