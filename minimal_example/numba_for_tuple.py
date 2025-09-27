import numpy as np
from numba import njit, types, literal_unroll
from numba.typed import List, Dict
from numba.core.types import unicode_type
from enum import IntEnum


# 定义 Numba 类型别名
nb_float = types.float64


class MaxIndicatorCount(IntEnum):
    sma = 3
    ema = 3
    bbands = 1
    rsi = 1
    atr = 1
    psar = 1


mic = MaxIndicatorCount

enable_cache = True


@njit(cache=enable_cache)
def populate_indicator_dicts(num, value_list, optim_list, all_indicator_data):
    for i in range(num + 1):
        value_dict = Dict.empty(unicode_type, types.float64)
        optim_dict = Dict.empty(unicode_type, types.float64)
        value_list.append(value_dict)
        optim_list.append(optim_dict)

    for param_tuple in all_indicator_data:
        _n, _n2, _s, s2, arr = param_tuple

        _value_dict = value_list[_n]
        _optim_dict = optim_list[_n]

        if s2 == "enable":
            if len(arr) >= 1:
                _value_dict[f"{_s}_{s2}_{_n2}"] = arr[0]
            if len(arr) >= 2:
                _optim_dict[f"{_s}_{'optim'}_{_n2}"] = arr[1]
        elif len(arr) == 4:
            _value_dict[f"{_s}_{s2}_{_n2}"] = arr[0]
            _optim_dict[f"{_s}_{s2}_min_{_n2}"] = arr[1]
            _optim_dict[f"{_s}_{s2}_max_{_n2}"] = arr[2]
            _optim_dict[f"{_s}_{s2}_step_{_n2}"] = arr[3]


@njit(cache=enable_cache)
def test_nested_tuple_nparray_support():
    dict_type = Dict.empty(unicode_type, types.float64)
    value_list = List.empty_list(dict_type)
    dict_type2 = Dict.empty(unicode_type, types.float64)
    optim_list = List.empty_list(dict_type2)

    num = 0
    bbands = (
        (num, 0, "bbands", "enable", np.array([True, True], dtype=np.float64)),
        (num, 0, "bbands", "period", np.array([14, 5, 50, 1], dtype=np.float64)),
        (num, 0, "bbands", "std_mult", np.array([2, 1, 5, 0.5], dtype=np.float64)),
    )
    num += 1
    sma = (
        (num, 0, "sma", "enable", np.array([True, True], dtype=np.float64)),
        (num, 0, "sma", "period", np.array([14, 6, 200, 4], dtype=np.float64)),
        # SMA 1
        (num, 1, "sma", "enable", np.array([True, True], dtype=np.float64)),
        (num, 1, "sma", "period", np.array([200, 100, 40, 5], dtype=np.float64)),
        # # SMA 2
        # (num, 2, "sma", "enable", np.array([True, True], dtype=np.float64)),
        # (num, 2, "sma", "period", np.array([200, 100, 40, 3], dtype=np.float64)),
    )
    all_indicator_data = (*bbands, *sma)

    populate_indicator_dicts(num, value_list, optim_list, all_indicator_data)

    return value_list, optim_list


# --- 运行测试 ---
if __name__ == "__main__":
    print("--- 正在测试 Numba 对嵌套异构元组的支持 nparray版本 ---")
    # Numba 会在这一步进行编译
    result = test_nested_tuple_nparray_support()
    print(f"结果 value_list: {result[0]}")
    print(f"结果 optim_list: {result[1]}")
