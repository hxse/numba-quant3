from numba import njit, types
from numba.typed import Dict
import numpy as np
from enum import IntEnum


class MaxIndicatorCount(IntEnum):
    sma = 3
    ema = 3
    bbands = 3
    rsi = 1
    atr = 1
    psar = 1


@njit
def test_fstring_in_numba():
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )

    for i in range(MaxIndicatorCount.sma.value):
        key = f"sma_enable" + f"{i}"
        params[key] = i

    return params


@njit
def test_format_in_numba():
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )

    for i in range(MaxIndicatorCount.ema.value):
        key = "ema_enable_{}".format(i)
        params[key] = i

    return params


# --- 运行测试 ---
if __name__ == "__main__":
    try:
        result = test_fstring_in_numba()  # 测试会通过
        print("测试成功 (这意味着您的 Numba 版本可能支持 f-string)")
        print(result)
    except Exception as e:
        print("\n测试失败：Numba 不支持 f-string。")
        print("错误类型:", type(e).__name__)

    try:
        result = test_format_in_numba()  # 测试不会通过
        print("测试成功 (这意味着您的 Numba 版本可能支持 format)")
        print(result)
    except Exception as e:
        print("\n测试失败：Numba 不支持 format。")
        print("错误类型:", type(e).__name__)
