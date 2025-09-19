from numba import njit
from numba.typed import Dict, List
from numba.core import types
import numpy as np

from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_float = numba_config["nb"]["float"]


@njit(cache=enable_cache)
def get_dict_keys_as_list(params_dict):
    """
    将 Numba 字典的键转换为 Numba List。
    """
    keys = List.empty_list(types.unicode_type)
    for i in params_dict.keys():
        keys.append(i)
    return keys


@njit(nb_float[:](types.DictType(types.unicode_type, nb_float)), cache=enable_cache)
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
