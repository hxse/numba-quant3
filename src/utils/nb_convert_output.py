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


# 需要重载的函数
def get_item_from_dict_list(data_list, num):
    pass


@overload(get_item_from_dict_list, jit_options={"cache": cache})
def ov_get_item_from_dict_list(data_list, num):
    # 验证输入类型是否为列表，且列表元素是字典，且索引是整数
    if not (
        isinstance(data_list, types.ListType)
        and isinstance(data_list.dtype, types.DictType)
        and isinstance(num, types.Integer)
    ):
        return None

    # 获取字典中的值类型
    value_type = data_list.dtype.value_type

    # 定义通用的实现函数
    def create_impl(dtype):
        def impl(data_list, num):
            if num < 0 or num >= len(data_list):
                return Dict.empty(types.unicode_type, dtype)
            return data_list[num]

        return impl

    # --- 使用 if/elif 结构进行精简 ---
    if value_type == nb_float[:]:
        return create_impl(value_type)
    elif value_type == nb_bool[:]:
        return create_impl(value_type)
    elif value_type == nb_int[:]:
        return create_impl(value_type)
    elif value_type == nb_float:
        return create_impl(value_type)
    elif value_type == nb_bool:
        return create_impl(value_type)
    elif value_type == nb_int:
        return create_impl(value_type)

    return None


# --- 重构 convert_dict_to_2d_array 系列函数 ---
def convert_dict_to_np_array(params_dict):
    pass


@overload(convert_dict_to_np_array, jit_options={"cache": cache})
def ov_convert_dict_to_2d_array(params_dict):
    # 将重复的检查提取到最前面
    if not (
        isinstance(params_dict, types.DictType)
        and params_dict.key_type == types.unicode_type
    ):
        return None

    # 定义核心实现逻辑 (用于数组值)
    @njit(parallel=True, cache=cache)
    def convert_impl_array(params_dict, dtype):
        keys = get_dict_keys_as_list(params_dict)
        num_keys = len(keys)

        if num_keys == 0:
            return np.empty((0, 0), dtype=dtype)

        # 获取第一个键
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

    # 定义核心实现逻辑 (用于标量值)
    @njit(parallel=True, cache=cache)
    def convert_impl_scalar(params_dict, dtype):
        keys = get_dict_keys_as_list(params_dict)
        num_keys = len(keys)

        if num_keys == 0:
            return np.empty((0), dtype=dtype)

        shape = (num_keys,)
        result_array = np.empty(shape, dtype=dtype)

        for i in prange(num_keys):
            _i = nb_int(i)
            key = keys[_i]
            result_array[_i] = params_dict[key]

        return result_array

    # 提取公共函数来创建具体的实现
    def create_impl_factory(impl_func, dtype):
        def impl(params_dict):
            return impl_func(params_dict, dtype)

        return impl

    # 检查字典的值类型并返回相应的实现
    value_type = params_dict.value_type

    # --- 数组模式 ---
    if value_type == nb_int[:]:
        return create_impl_factory(convert_impl_array, nb_int)
    elif value_type == nb_float[:]:
        return create_impl_factory(convert_impl_array, nb_float)
    elif value_type == nb_bool[:]:
        return create_impl_factory(convert_impl_array, nb_bool)

    # --- 标量模式 ---
    elif value_type == nb_int:
        return create_impl_factory(convert_impl_scalar, nb_int)
    elif value_type == nb_float:
        return create_impl_factory(convert_impl_scalar, nb_float)
    elif value_type == nb_bool:
        return create_impl_factory(convert_impl_scalar, nb_bool)

    return None


# 封装所有转换逻辑的函数
@njit(cache=cache)
def jitted_convert_all_dicts(
    params_list,
    result_list,
    num,
):
    """
    在一个 JIT 函数内，根据类型和指定索引调用特定的转换函数。
    """

    (
        tohlcv,
        indicator_params_list,
        backtest_params_list,
        tohlcv_mtf,
        indicator_params_list_mtf,
        mapping_mtf,
        tohlcv_smoothed,
        tohlcv_mtf_smoothed,
        is_only_performance,
    ) = params_list

    (
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_mtf,
    ) = result_list

    tohlcv_dict = tohlcv
    tohlcv_mtf_dict = tohlcv_mtf
    mapping_mtf_dict = mapping_mtf
    tohlcv_smoothed_dict = tohlcv_smoothed
    tohlcv_mtf_smoothed_dict = tohlcv_mtf_smoothed

    # 从索引num中提取item
    indicator_params_dict = get_item_from_dict_list(indicator_params_list, num)
    backtest_params_dict = get_item_from_dict_list(backtest_params_list, num)
    indicator_params_mtf_dict = get_item_from_dict_list(indicator_params_list_mtf, num)

    # 把key提取成list
    indicator_params_keys = get_dict_keys_as_list(indicator_params_dict)
    backtest_params_keys = get_dict_keys_as_list(backtest_params_dict)
    indicator_params_mtf_keys = get_dict_keys_as_list(indicator_params_mtf_dict)

    # 把字典提取成1d数组
    indicator_params_np = convert_dict_to_np_array(indicator_params_dict)
    backtest_params_np = convert_dict_to_np_array(backtest_params_dict)
    indicator_params_mtf_np = convert_dict_to_np_array(indicator_params_mtf_dict)

    # 把key提取成list
    tohlcv_keys = get_dict_keys_as_list(tohlcv)
    tohlcv_mtf_keys = get_dict_keys_as_list(tohlcv_mtf)
    mapping_mtf_keys = get_dict_keys_as_list(mapping_mtf)
    tohlcv_smoothed_keys = get_dict_keys_as_list(tohlcv_smoothed)
    tohlcv_mtf_smoothed_keys = get_dict_keys_as_list(tohlcv_mtf_smoothed)

    # 把字典提取成2d数组
    tohlcv_np = convert_dict_to_np_array(tohlcv_dict)
    tohlcv_mtf_np = convert_dict_to_np_array(tohlcv_mtf_dict)
    mapping_mtf_np = convert_dict_to_np_array(mapping_mtf_dict)
    tohlcv_smoothed_np = convert_dict_to_np_array(tohlcv_smoothed_dict)
    tohlcv_mtf_smoothed_np = convert_dict_to_np_array(tohlcv_mtf_smoothed_dict)

    # 从索引num中提取item
    indicators_dict = get_item_from_dict_list(indicators_output_list, num)
    signals_dict = get_item_from_dict_list(signals_output_list, num)
    backtest_dict = get_item_from_dict_list(backtest_output_list, num)
    indicators_mtf_dict = get_item_from_dict_list(indicators_output_list_mtf, num)

    indicators_keys = get_dict_keys_as_list(indicators_dict)
    signals_keys = get_dict_keys_as_list(signals_dict)
    backtest_keys = get_dict_keys_as_list(backtest_dict)
    indicators_mtf_keys = get_dict_keys_as_list(indicators_mtf_dict)

    # 把字典提取成2d数组
    indicators_np = convert_dict_to_np_array(indicators_dict)
    signals_np = convert_dict_to_np_array(signals_dict)
    backtest_np = convert_dict_to_np_array(backtest_dict)
    indicators_mtf_np = convert_dict_to_np_array(indicators_mtf_dict)

    performance_dict = get_item_from_dict_list(performance_output_list, num)
    performance_keys = get_dict_keys_as_list(performance_dict)
    # 把字典提取成1d数组
    # performance_np = get_dict_values_as_np_array(performance_dict)
    performance_np = convert_dict_to_np_array(performance_dict)

    return (
        ("tohlcv", tohlcv_keys, tohlcv_dict, tohlcv_np),
        ("tohlcv_mtf", tohlcv_mtf_keys, tohlcv_mtf_dict, tohlcv_mtf_np),
        ("mapping_mtf", mapping_mtf_keys, mapping_mtf_dict, mapping_mtf_np),
        (
            "tohlcv_smoothed",
            tohlcv_smoothed_keys,
            tohlcv_smoothed_dict,
            tohlcv_smoothed_np,
        ),
        (
            "tohlcv_mtf_smoothed",
            tohlcv_mtf_smoothed_keys,
            tohlcv_mtf_smoothed_dict,
            tohlcv_mtf_smoothed_np,
        ),
        #
        (
            "indicator_params",
            indicator_params_keys,
            indicator_params_dict,
            indicator_params_np,
        ),
        (
            "backtest_params",
            backtest_params_keys,
            backtest_params_dict,
            backtest_params_np,
        ),
        (
            "indicator_params_mtf",
            indicator_params_mtf_keys,
            indicator_params_mtf_dict,
            indicator_params_mtf_np,
        ),
        #
        ("indicators", indicators_keys, indicators_dict, indicators_np),
        ("signals", signals_keys, signals_dict, signals_np),
        ("backtest", backtest_keys, backtest_dict, backtest_np),
        ("indicators_mtf", indicators_mtf_keys, indicators_mtf_dict, indicators_mtf_np),
        #
        ("performance", performance_keys, performance_dict, performance_np),
    )
