import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict, List

# 从全局配置中获取 Numba 类型
from src.utils.constants import numba_config

cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]
np_float = numba_config["np"]["float"]

print("params cache", cache)


@njit(cache=cache)
def get_indicator_params():
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    params["sma_enable"] = nb_float(0)
    params["sma_period"] = nb_float(14)

    params["sma2_enable"] = nb_float(0)
    params["sma2_period"] = nb_float(14)

    params["bbands_enable"] = nb_float(0)
    params["bbands_period"] = nb_float(14)
    params["bbands_std_mult"] = nb_float(2.0)
    return params


@njit(cache=cache)
def get_backtest_params():
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    params["signal_select"] = nb_float(0)
    params["atr_sl_mult"] = nb_float(2.0)
    return params


@njit(cache=cache)
def create_params_list_template(params_count):
    assert params_count >= 0, "参数组合数量必须大于等于0"
    indicator_params_list = List.empty_list(
        Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
    )
    backtest_params_list = List.empty_list(
        Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
    )

    for n in range(params_count):
        indicator_params_list.append(get_indicator_params())
        backtest_params_list.append(get_backtest_params())

    return (indicator_params_list, backtest_params_list)


@njit(cache=cache)
def create_params_dict_template(params_count):
    """
    根据numba文档, 目前只支持List里面放Dict, 不支持Dict里面放List
    所以,把List转成numpy数组解决问题
    """

    assert params_count > 0, "参数组合数量必须大于0"

    indicator_params_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float[:],  # 使用 NumPy 数组类型
    )

    backtest_params_dict = Dict.empty(
        key_type=types.unicode_type, value_type=nb_float[:]
    )

    params = get_indicator_params()
    for key in params.keys():
        # 直接创建 NumPy 数组，填充相同的值
        arr = np.zeros(params_count, dtype=nb_float)
        arr[:] = params[key]
        indicator_params_dict[key] = arr

    params = get_backtest_params()
    for key in params.keys():
        # 直接创建 NumPy 数组，填充相同的值
        arr = np.zeros(params_count, dtype=nb_float)
        arr[:] = params[key]
        backtest_params_dict[key] = arr

    return (indicator_params_dict, backtest_params_dict)


@njit(cache=cache)
def get_params_list_value(key, params_list):
    params_count = len(params_list)
    arr = np.zeros(params_count, dtype=nb_float)

    for i in range(params_count):
        arr[i] = params_list[i][key]
    return arr


@njit(cache=cache)
def set_params_list_value(key, params_list, arr):
    params_count = len(params_list)
    assert params_count == len(arr), (
        f"更新数量应该和原始数量一致{params_count} {len(arr)}"
    )

    for i in range(params_count):
        if i == "":
            continue
        params_list[i][key] = arr[i]


@njit(cache=cache)
def get_params_dict_value(key: str, params_dict):
    return params_dict[key]


@njit(cache=cache)
def set_params_dict_value(key: str, params_dict, arr: np.ndarray):
    first_key = ""
    for k in params_dict.keys():
        first_key = k
        break

    if first_key:
        params_count = len(params_dict[first_key])
        assert params_count == len(arr), "更新数量应该和原始数量一致"

    params_dict[key] = arr


@njit(cache=cache)
def convert_params_dict_list(params_dict):
    first_key = ""
    for k in params_dict.keys():
        first_key = k
        break

    # 检查字典是否为空
    if not first_key:
        return List.empty_list(
            Dict.empty(
                key_type=types.unicode_type,
                value_type=nb_float,
            )
        )

    params_count = len(params_dict[first_key])

    for i in params_dict.keys():
        assert params_count == len(params_dict[i]), "参数数量要彼此一致"

    params_list = List.empty_list(
        Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
    )

    for i in range(params_count):
        params = Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
        for key in params_dict.keys():
            params[key] = params_dict[key][i]
        params_list.append(params)

    return params_list


@njit(cache=cache)
def get_data_mapping(tohlcv_np, tohlcv_np_mtf):
    _d = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float[:],
    )
    _d["mtf"] = np.zeros(tohlcv_np.shape[0], dtype=nb_float)
    return _d
