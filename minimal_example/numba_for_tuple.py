import numpy as np
from numba import njit, types, literal_unroll
from numba.typed import List, Dict
from numba.core.types import unicode_type


# 假设 MaxIndicatorCount 和 mic 已在文件顶部定义
# 我们可以将其替换为常量以便独立运行
class MockMic:
    @property
    def bbands(self):
        # 只需要一个大于等于 1 的值
        class Value:
            value = 1

        return Value()


mic = MockMic()


@njit
def test_nested_tuple_nparray_support():
    dict_type = Dict.empty(unicode_type, types.float64)
    value_list = List.empty_list(dict_type)
    dict_type2 = Dict.empty(unicode_type, types.float64)
    optim_list = List.empty_list(dict_type2)

    num = 0
    bbands = (
        (num, 0, "bbands", "enable", np.array([True, True], dtype=np.float64)),
        (num, 0, "bbands", "period", np.array([14, 5, 50, 1], dtype=np.float64)),
        (num, 0, "bbands", "std_mult", np.array([2, 1, 4, 0.5], dtype=np.float64)),
    )
    num += 1
    sma = (
        (num, 0, "sma", "enable", np.array([True, True], dtype=np.float64)),
        (num, 0, "sma", "period", np.array([14, 10, 200, 5], dtype=np.float64)),
        # SMA 1
        (num, 1, "sma", "enable", np.array([True, True], dtype=np.float64)),
        (num, 1, "sma", "period", np.array([200, 100, 40, 10], dtype=np.float64)),
    )
    all_indicator_data = (*bbands, *sma)

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

    return value_list, optim_list


# --- 运行测试 ---
if __name__ == "__main__":
    print("--- 正在测试 Numba 对嵌套异构元组的支持 nparray版本 ---")
    # Numba 会在这一步进行编译
    result = test_nested_tuple_nparray_support()
    print(f"结果 value_list: {result[0]}")
    print(f"结果 optim_list: {result[1]}")
