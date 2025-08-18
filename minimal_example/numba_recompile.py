# test_recompile.py
from numba import njit, types
from numba.typed import List
import time

# 全局整数常量
# 尝试修改这个值（例如从 10 改为 20）
GLOBAL_INT = 10

# 全局字符串元组
# 尝试修改这个元组（例如添加一个元素 "str4"）
GLOBAL_TUPLE_STR = ("str1", "str2", "str3")

print(f"当前 GLOBAL_INT: {GLOBAL_INT}")
print(f"当前 GLOBAL_TUPLE_STR: {GLOBAL_TUPLE_STR}")


@njit(cache=True)
def test_recompile_behavior():
    """
    一个使用全局变量的 Numba 函数
    """
    total_len = 0
    # 在 Numba 中使用全局元组
    for s in GLOBAL_TUPLE_STR:
        total_len += len(s)

    # 在 Numba 中使用全局整数
    result = total_len + GLOBAL_INT

    return result


def main():
    print("-" * 20)
    print("开始调用 Numba 函数...")
    start_time = time.time()

    # 第一次调用，会触发编译或从缓存加载
    result = test_recompile_behavior()

    end_time = time.time()
    print(f"函数调用结果: {result}")
    print(f"函数调用耗时: {end_time - start_time:.4f} 秒")
    print("-" * 20)


if __name__ == "__main__":
    main()
