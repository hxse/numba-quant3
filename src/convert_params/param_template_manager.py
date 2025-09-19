import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict, List

from src.convert_params.param_template import get_backtest_params, get_indicator_params
from src.convert_params.nb_params_signature import (
    create_params_list_template_signature,
    create_params_dict_template_signature,
    get_params_list_value_signature,
    set_params_list_value_signature,
    get_params_dict_value_signature,
    set_params_dict_value_signature,
)


from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_float = numba_config["nb"]["float"]


@njit(create_params_list_template_signature, cache=enable_cache)
def create_params_list_template(params_count, empty):
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
        indicator_params_list.append(get_indicator_params(empty))
        backtest_params_list.append(get_backtest_params(empty))

    return (indicator_params_list, backtest_params_list)


@njit(create_params_dict_template_signature, cache=enable_cache)
def create_params_dict_template(params_count, empty):
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

    params = get_indicator_params(empty)
    for key in params.keys():
        # 直接创建 NumPy 数组，填充相同的值
        arr = np.zeros(params_count, dtype=nb_float)
        arr[:] = params[key]
        indicator_params_dict[key] = arr

    params = get_backtest_params(empty)
    for key in params.keys():
        # 直接创建 NumPy 数组，填充相同的值
        arr = np.zeros(params_count, dtype=nb_float)
        arr[:] = params[key]
        backtest_params_dict[key] = arr

    return (indicator_params_dict, backtest_params_dict)


@njit(get_params_list_value_signature, cache=enable_cache)
def get_params_list_value(key, params_list):
    params_count = len(params_list)
    arr = np.zeros(params_count, dtype=nb_float)

    for i in range(params_count):
        arr[i] = params_list[i][key]
    return arr


@njit(set_params_list_value_signature, cache=enable_cache)
def set_params_list_value(key, params_list, arr):
    params_count = len(params_list)
    assert params_count == len(arr), (
        f"更新数量应该和原始数量一致{params_count} {len(arr)}"
    )

    for i in range(params_count):
        if i == "":
            continue
        params_list[i][key] = arr[i]


@njit(get_params_dict_value_signature, cache=enable_cache)
def get_params_dict_value(key: str, params_dict):
    return params_dict[key]


@njit(set_params_dict_value_signature, cache=enable_cache)
def set_params_dict_value(key: str, params_dict, arr: np.ndarray):
    first_key = ""
    for k in params_dict.keys():
        first_key = k
        break

    if first_key:
        params_count = len(params_dict[first_key])
        assert params_count == len(arr), "更新数量应该和原始数量一致"

    params_dict[key] = arr
