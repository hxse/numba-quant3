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


# -----------------------------------------------------------------
# 辅助函数: 用于更新 value_list 和 optim_list
# -----------------------------------------------------------------
@njit(cache=enable_cache)
def update_value_optim_dicts(
    num, value_list, optim_list, list_0, list_1, list_2, list_3, list_4
):
    """
    根据分离的列表更新指标的 value 和 optim 字典。

    由于 Numba 不支持将分离的列表重新 zip/解压成原始元组，
    我们通过同时迭代五个列表的索引来实现原始的逻辑。
    """

    # 第二部分：初始化 value_list 和 optim_list
    for i in range(num + 1):
        value_dict = Dict.empty(unicode_type, nb_float)
        optim_dict = Dict.empty(unicode_type, nb_float)
        value_list.append(value_dict)
        optim_list.append(optim_dict)

    # 假设所有列表长度一致，使用 list_0 的长度
    for i in range(len(list_0)):
        # 模拟原始的 param_tuple 解包: _n, _n2, _s, _s2, arr = param_tuple
        _n = list_0[i]
        _n2 = list_1[i]
        _s = list_2[i]
        _s2 = list_3[i]
        arr = list_4[i]

        _value_dict = value_list[int(_n)]  # _n 是 float，需要转 int 作为索引
        _optim_dict = optim_list[int(_n)]

        # Numba 支持 f-string
        if _s2 == "enable":
            if len(arr) >= 1:
                _value_dict[f"{_s}_{_s2}_{_n2}"] = arr[0]
            if len(arr) >= 2:
                _optim_dict[f"{_s}_{'optim'}_{_n2}"] = arr[1]
        elif len(arr) == 4:
            _value_dict[f"{_s}_{_s2}_{_n2}"] = arr[0]
            _optim_dict[f"{_s}_{_s2}_min_{_n2}"] = arr[1]
            _optim_dict[f"{_s}_{_s2}_max_{_n2}"] = arr[2]
            _optim_dict[f"{_s}_{_s2}_step_{_n2}"] = arr[3]

    # 返回更新后的列表（虽然是原地修改，但返回是好的习惯）
    return value_list, optim_list


# -----------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------


@njit(cache=enable_cache)
def test_nested_tuple_nparray_support():
    dict_type = Dict.empty(unicode_type, nb_float)
    value_list = List.empty_list(dict_type)
    dict_type2 = Dict.empty(unicode_type, nb_float)
    optim_list = List.empty_list(dict_type2)

    num = 0
    bbands_params = (
        (num, 0, "bbands", "enable", np.array([True, True], dtype=nb_float)),
        (num, 0, "bbands", "period", np.array([14, 5, 50, 1], dtype=nb_float)),
        (num, 0, "bbands", "std_mult", np.array([2, 1, 4, 0.5], dtype=nb_float)),
    )
    # 假设 mic 在全局范围内可用
    assert bbands_params[-1][1] < mic.bbands.value, (
        f"bbands数量超出最大限制 {bbands_params[-1][1]} {mic.bbands.value}"
    )
    num += 1
    sma_params = (
        (num, 0, "sma", "enable", np.array([True, True], dtype=nb_float)),
        (num, 0, "sma", "period", np.array([14, 10, 200, 5], dtype=nb_float)),
        # SMA 1
        (num, 1, "sma", "enable", np.array([True, True], dtype=nb_float)),
        (num, 1, "sma", "period", np.array([200, 100, 40, 10], dtype=nb_float)),
    )
    # 假设 mic 在全局范围内可用
    assert sma_params[-1][1] < mic.sma.value, (
        f"sma数量超出最大限制 {sma_params[-1][1]} {mic.sma.value}"
    )
    all_indicator_tuple = (*bbands_params, *sma_params)

    list_0 = List([p[0] for p in all_indicator_tuple])  # nb_float
    list_1 = List([p[1] for p in all_indicator_tuple])  # nb_float
    list_2 = List([p[2] for p in all_indicator_tuple])  # unicode_type
    list_3 = List([p[3] for p in all_indicator_tuple])  # unicode_type
    list_4 = List([p[4] for p in all_indicator_tuple])  # nb_float[:]

    value_list, optim_list = update_value_optim_dicts(
        num, value_list, optim_list, list_0, list_1, list_2, list_3, list_4
    )

    return value_list, optim_list


# --- 运行测试 ---
if __name__ == "__main__":
    print("--- 正在测试 Numba 对嵌套异构元组的支持 nparray版本 ---")
    value_list, optim_list = test_nested_tuple_nparray_support()
    print("value_list", value_list)
    print("optim_list", optim_list)
