import numpy as np
import numba as nb

# --- 成功案例：Numba 可以推断多种类型 ---


@nb.njit
def process_data(data):
    # Numba 看到 "is None" 就会知道 data 参数可能为 None
    # 于是它会推断 data 的类型为 "Optional(ndarray)"
    if data is None:
        print("传入 None，创建一个新的数组。")
        return np.arange(3)
    else:
        print("传入数组，对它进行操作。")
        return data * 2


# --- 失败案例：Numba 无法推断类型，因为没有“提示” ---


@nb.njit
def process_data_no_hint(data):
    # Numba 在编译时不知道 data 可能为 None
    # 它会假定 data 是一种支持乘法的类型，比如 ndarray
    # 但当你传入 None 时，就会出错
    return data * 2


# --- 运行测试 ---

if __name__ == "__main__":
    print("--- 测试成功案例 ---")

    # 第一次调用：传入 None
    # Numba 在编译时会处理 None 这种可能性
    result1 = process_data(None)
    print("返回结果:", result1)

    # 第二次调用：传入 ndarray
    # 由于 Numba 已经推断出 data 可以是 ndarray，所以这次调用没有问题
    test_array = np.array([10, 20, 30])
    result2 = process_data(test_array)
    print("返回结果:", result2)

    # 第三次调用：传入 int
    # 导致 Numba 编译错误。因为函数的一个分支返回 np.ndarray，
    # 而这个分支返回 int。Numba 要求函数的所有返回值类型必须一致。
    test_int = 1
    try:
        print(f"尝试用 int ({test_int}) 调用 process_data...")
        result_int = process_data(test_int)
        print("返回结果:", result_int)
    except nb.errors.TypingError as e:
        print("捕获到 Numba 编译错误！")
        print("错误信息:", e)

    print("\n" + "=" * 40 + "\n")
    print("--- 测试失败案例 ---")

    try:
        # 这个调用会失败，因为它没有处理 None 的逻辑
        print("尝试用 None 调用 process_data_no_hint...")
        process_data_no_hint(None)
    except nb.errors.TypingError as e:
        print("捕获到 Numba 编译错误！")
        print("错误信息:", e)
