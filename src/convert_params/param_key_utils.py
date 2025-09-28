import numpy as np
import numba as nb
from numba import njit, typeof, literal_unroll
from numba.core import types
from numba.core.types import unicode_type
from numba.typed import Dict, List
from src.convert_output.nb_dict_to_array_converter import (
    get_dict_keys,
    get_dict_values,
    get_dict_keys_and_values,
    get_dict_keys_wrapper,
    get_dict_values_wrapper,
    get_dict_keys_and_values_wrapper,
)
from src.convert_params.numba_constructors import (
    create_list_unicode_empty,
    create_list_unicode_one,
    create_list_dict_float_1d_empty,
    create_list_dict_float_1d_one,
    create_2d_list_unicode_empty,
    create_2d_list_unicode_one,
    create_dict_float_1d_empty,
    create_dict_float_1d_one,
)

# 从全局配置中获取 Numba 类型
from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]
np_float = numba_config["np"]["float"]


# --- 更新后的 JIT 工具函数 ---
@njit(cache=enable_cache)
def get_length_from_list_or_dict(data_list):
    """
    一个JIT辅助函数，用于从Numba List中取出指定索引的元素。
    """
    return len(data_list)


# --- 更新后的 JIT 工具函数 ---
@njit(cache=enable_cache)
def get_item_from_list(data_list, num):
    """
    一个JIT辅助函数，用于从Numba List中取出指定索引的元素。
    """
    assert 0 <= num < len(data_list), "Index out of bounds for Numba List"
    return data_list[num]


@njit(cache=enable_cache)
def get_item_from_2d_list(data_array, row_num, col_num):
    """
    一个JIT辅助函数，用于从二维NumPy数组中取出指定索引的元素。

    参数:
    data_array numba List 2d。
    row_num (int): 行索引。
    col_num (int): 列索引。

    返回:
    float: 数组在 (row_num, col_num) 位置上的元素值。
    """
    # 确保行索引在数组的边界内
    assert 0 <= row_num < len(data_array), "Row index out of bounds"

    # 确保列索引在指定行的边界内
    # 这种写法在Numba中可以，但在标准的NumPy二维数组中是多余的
    assert 0 <= col_num < len(data_array[row_num]), "Column index out of bounds"

    return data_array[row_num][col_num]


@njit(cache=enable_cache)
def append_item(_list, item):
    _list.append(item)


@njit(cache=enable_cache)
def get_item_from_dict(data_dict, key):
    assert key in data_dict, "key不在dict之中"
    return data_dict[key]


def convert_nb_list_to_py_list(nb_list):
    return [
        get_item_from_list(nb_list, i)
        for i in range(get_length_from_list_or_dict(nb_list))
    ]


def get_nb_dict_keys_as_py_list(nb_dict):
    keys = get_dict_keys_wrapper(nb_dict)
    keys = convert_nb_list_to_py_list(keys)
    return keys


def get_nb_dict_value_as_py_list(nb_dict):
    values = get_dict_values_wrapper(nb_dict)
    values = convert_nb_list_to_py_list(values)
    return values


def get_nb_dict_keys_and_value_as_py_list(nb_dict):
    keys, values = get_dict_keys_and_values_wrapper(nb_dict)
    keys = convert_nb_list_to_py_list(keys)
    values = convert_nb_list_to_py_list(values)
    return keys, values


def get_nb_dict_keys_as_py_dict(nb_dict):
    keys, values = get_nb_dict_keys_and_value_as_py_list(nb_dict)
    return {k: v for k, v in zip(keys, values)}
