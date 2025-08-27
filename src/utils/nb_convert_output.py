from numba import njit, prange
from numba.typed import Dict, List
from numba.core import types
import numpy as np
from numba.extending import overload

from src.utils.constants import numba_config

cache = numba_config["cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


# 辅助函数，将字典键转换为 List，保持不变
@njit(cache=cache)
def get_dict_keys_as_list(params_dict):
    """
    将 Numba 字典的键转换为 Numba List。
    """
    keys = List.empty_list(types.unicode_type)
    for i in params_dict.keys():
        keys.append(i)
    return keys


@njit(nb_float[:](types.DictType(types.unicode_type, nb_float)), cache=cache)
def get_dict_values_as_np_array(params_dict):
    """
    将 Numba 字典中的所有值提取到一个一维 NumPy 数组中。

    参数:
        params_dict (DictType(unicode_type, nb_float)):
            Numba 字典，键为字符串，值为 nb_float。

    返回:
        np.array: 包含所有值的 NumPy 一维数组。
    """
    num_items = len(params_dict)

    # 如果字典为空，返回一个空的 NumPy 数组
    if num_items == 0:
        return np.empty(0, dtype=nb_float)

    result_array = np.empty(num_items, dtype=nb_float)

    # 使用 enumerate 遍历字典的值并赋值
    for i, value in enumerate(params_dict.values()):
        result_array[i] = value

    return result_array


# --- 重构 get_item_from_dict_list 系列函数 ---
def get_item_from_dict_list(data_list, num):
    pass


@overload(get_item_from_dict_list, jit_options={"cache": cache})
def ov_get_item_from_dict_list(data_list, num):
    # 检查列表中的字典值类型是否为 float 数组
    if (
        isinstance(data_list, types.ListType)
        and isinstance(data_list.dtype, types.DictType)
        and data_list.dtype.value_type == nb_float[:]
        and isinstance(num, types.Integer)
    ):

        def float_impl(data_list, num):
            if num < 0 or num >= len(data_list):
                return Dict.empty(types.unicode_type, types.float64[:])
            return data_list[num]

        return float_impl

    # 检查列表中的字典值类型是否为 bool 数组
    if (
        isinstance(data_list, types.ListType)
        and isinstance(data_list.dtype, types.DictType)
        and data_list.dtype.value_type == nb_bool[:]
        and isinstance(num, types.Integer)
    ):

        def bool_impl(data_list, num):
            if num < 0 or num >= len(data_list):
                return Dict.empty(types.unicode_type, types.boolean[:])
            return data_list[num]

        return bool_impl

    if (
        isinstance(data_list, types.ListType)
        and isinstance(data_list.dtype, types.DictType)
        and data_list.dtype.value_type == nb_float
        and isinstance(num, types.Integer)
    ):

        def simple_float_impl(data_list, num):
            if num < 0 or num >= len(data_list):
                return Dict.empty(types.unicode_type, nb_float)
            return data_list[num]

        return simple_float_impl

    return None


# --- 重构 convert_dict_to_2d_array 系列函数 ---
def convert_dict_to_2d_array(params_dict):
    pass


@overload(convert_dict_to_2d_array, jit_options={"cache": cache})
def ov_convert_dict_to_2d_array(params_dict):
    # 定义核心实现逻辑
    @njit(parallel=True, cache=cache)
    def convert_impl(params_dict, dtype):
        keys = get_dict_keys_as_list(params_dict)
        num_keys = len(keys)

        if num_keys == 0:
            return np.empty((0, 0), dtype=dtype)

        # 用你的方式获取第一个键
        first_key = ""
        for k in keys:
            first_key = k
            break

        first_key_array_len = len(params_dict[first_key])

        shape = (first_key_array_len, num_keys)
        result_array = np.empty(shape, dtype=dtype)

        # 验证每个数组的长度是否一致
        for key in keys:
            assert len(params_dict[key]) == first_key_array_len

        for i in prange(num_keys):
            _i = nb_int(i)
            key = keys[_i]
            result_array[:, _i] = params_dict[key]

        return result_array

    # 检查字典的值类型是否是 float 数组
    if (
        isinstance(params_dict, types.DictType)
        and params_dict.key_type == types.unicode_type
        and params_dict.value_type == nb_float[:]
    ):

        def float_impl(params_dict):
            return convert_impl(params_dict, nb_float)

        return float_impl

    # 检查字典的值类型是否是 bool 数组
    if (
        isinstance(params_dict, types.DictType)
        and params_dict.key_type == types.unicode_type
        and params_dict.value_type == nb_bool[:]
    ):

        def bool_impl(params_dict):
            return convert_impl(params_dict, nb_bool)

        return bool_impl

    return None


# 封装所有转换逻辑的函数
@njit(cache=cache)
def jitted_convert_all_dicts(
    indicators_output_list,
    signals_output_list,
    backtest_output_list,
    performance_output_list,
    indicators_output_list_mtf,
    num,
):
    """
    在一个 JIT 函数内，根据类型和指定索引调用特定的转换函数。
    """
    indicators_dict = get_item_from_dict_list(indicators_output_list, num)
    signals_dict = get_item_from_dict_list(signals_output_list, num)
    backtest_dict = get_item_from_dict_list(backtest_output_list, num)
    indicators_dict_mtf = get_item_from_dict_list(indicators_output_list_mtf, num)

    indicators_keys = get_dict_keys_as_list(indicators_dict)
    signals_keys = get_dict_keys_as_list(signals_dict)
    backtest_keys = get_dict_keys_as_list(backtest_dict)
    indicators_keys_mtf = get_dict_keys_as_list(indicators_dict_mtf)

    # 将提取的字典转换为 NumPy 数组
    indicators_np = convert_dict_to_2d_array(indicators_dict)
    signals_np = convert_dict_to_2d_array(signals_dict)
    backtest_np = convert_dict_to_2d_array(backtest_dict)
    indicators_np_mtf = convert_dict_to_2d_array(indicators_dict_mtf)

    performance_dict = get_item_from_dict_list(performance_output_list, num)
    performance_keys = get_dict_keys_as_list(performance_dict)
    performance_value = get_dict_values_as_np_array(performance_dict)

    return (
        (indicators_keys, signals_keys, backtest_keys, indicators_keys_mtf),
        (indicators_dict, signals_dict, backtest_dict, indicators_dict_mtf),
        (
            indicators_np,
            signals_np,
            backtest_np,
            indicators_np_mtf,
        ),
        performance_keys,
        performance_dict,
        performance_value,
    )
