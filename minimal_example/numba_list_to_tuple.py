from numba import njit, types
from numba.typed import List


# 这是一个用于演示错误的函数，它不应该被成功编译
@njit
def convert_inside_jit(input_list):
    # 这一行会引发 Numba 编译器错误
    return tuple(input_list)


def run_error_test():
    """尝试运行错误示例并捕获异常。"""
    print("--- 尝试在JIT函数内部进行 Numba List 到 tuple 的转换 ---")
    try:
        # 创建一个 Numba List 作为输入
        my_numba_list = List.empty_list(types.int64)
        my_numba_list.append(10)
        my_numba_list.append(20)

        # 调用函数，这里会触发编译错误
        result = convert_inside_jit(my_numba_list)
        print(f"✅ 成功执行，结果: {result}")

    except Exception as e:
        print("❌ 失败！捕捉到以下异常：")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {e}")
        print("\n--- 错误分析 ---")
        print(
            "这个错误表明 Numba 的 nopython 模式无法处理 `tuple(numba.typed.List)` 操作。"
        )
        print("这种数据类型转换必须在 JIT 函数外部的 Python 解释器中完成。")


if __name__ == "__main__":
    run_error_test()
