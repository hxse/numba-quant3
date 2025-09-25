import numpy as np
from numba import njit
from numba.typed import Dict, List
from numba.core import types
from numba.core.types import ListType, DictType, Integer, unicode_type
from src.convert_output.nb_dict_to_array_converter import (
    merge_dict_to_np_array,
    merge_dict_to_np_array_wrapper,
)

from src.convert_params.param_key_utils import (
    get_length_from_list_or_dict,
    get_item_from_list,
    get_item_from_2d_list,
    get_item_from_dict,
    convert_nb_list_to_py_list,
    get_dict_keys,
    get_dict_keys_wrapper,
    get_nb_dict_keys_as_py_list,
    get_nb_dict_value_as_py_list,
    get_nb_dict_keys_and_value_as_py_list,
)

from src.utils.constants import numba_config


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


def assert_keys_values_length(keys, values):
    assert isinstance(keys, list), f"keys 参数类型错误，需要 list，实际为 {type(keys)}"
    assert isinstance(values, np.ndarray), (
        f"values 参数类型错误，需要 np.ndarray，实际为 {type(values)}"
    )

    _len_keys = len(keys)
    _values_shape_len = len(values.shape)

    if _values_shape_len == 1:
        # 如果是一维数组
        assert _len_keys == values.shape[0], "键的数量与一维数组的长度不匹配"
    elif _values_shape_len == 2:
        # 如果是二维数组
        assert _len_keys == values.shape[1], "键的数量与二维数组的列数不匹配"
    else:
        # 抛出异常以捕获任何其他意外维度
        raise ValueError("不支持的数组维度")


def merge_dict_wrapper(nb_dict_item):
    """
    将 Numba 字典转换为 NumPy 数组和 Python 字典，并进行数据验证。
    """
    # 将 Numba 字典转换为一个包含键和值的py元组。
    # 把nb的 Dict 对象转换成 元组(keys, values), keys是nb的List, values是nparray
    # 例如：{'a': 1, 'b': 2} 转换成 (['a', 'b'], array([1, 2]))
    keys, values = merge_dict_to_np_array_wrapper(nb_dict_item)

    # Numba List 类型无法直接在 Python 代码中操作，需要转换为 Python list。
    keys = convert_nb_list_to_py_list(keys)

    # 对转换后的 keys 和 values 进行形状和长度验证。
    # 这个函数在 Python 原生环境中运行，可以利用灵活的断言和错误处理。
    assert_keys_values_length(keys, values)

    # 即每一列代表一个字典的键, 二维数组需要对数组进行转置（values.T）
    # 对于一维数组，转置操作没有效果，代码依然安全。
    # 返回一个 Python 字典, key一般是str, value一般是np数组或标量
    return {key: val for key, val in zip(keys, values.T)}


def convert_nb_data_to_py_dicts(params_list, result_list, num):
    (
        ohlcv_mtf,
        ohlcv_smoothed_mtf,
        data_mapping,
        indicator_params_mtf,
        backtest_params,
        is_only_performance,
    ) = params_list

    (
        indicators_output_mtf,
        signals_output,
        backtest_output,
        performance_output,
    ) = result_list

    result = {}

    # 将所有需要处理的 Numba 列表存储在一个字典中
    data_to_process = {
        "ohlcv_mtf": ohlcv_mtf,
        "ohlcv_smoothed_mtf": ohlcv_smoothed_mtf,
        "backtest_params": backtest_params,
        "signals_output": signals_output,
        "backtest_output": backtest_output,
        "performance_output": performance_output,
    }

    # 遍历字典，处理单层列表
    for key, value in data_to_process.items():
        result[key] = [merge_dict_wrapper(i) for i in convert_nb_list_to_py_list(value)]

    # 处理二维嵌套列表
    nested_data_to_process = {
        "indicator_params_mtf": indicator_params_mtf,
        "indicators_output_mtf": indicators_output_mtf,
    }

    for key, value in nested_data_to_process.items():
        result[key] = [
            [merge_dict_wrapper(m) for m in convert_nb_list_to_py_list(i)]
            for i in convert_nb_list_to_py_list(value)
        ]

    # data_mapping 的处理是独特的，不能与其他代码合并
    result["data_mapping"] = {}
    keys, values = get_nb_dict_keys_and_value_as_py_list(data_mapping)
    for k, v in zip(keys, values):
        result["data_mapping"][k] = v

    return result
