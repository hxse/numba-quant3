import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict, List

from src.convert_params.nb_params_signature import (
    convert_params_dict_list_signature,
    convert_params_list_dict_signature,
)

from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_float = numba_config["nb"]["float"]


@njit(convert_params_dict_list_signature, cache=enable_cache)
def convert_params_dict_list(params_dict):
    first_key = ""
    for k in params_dict.keys():
        first_key = k
        break

    params_list = List.empty_list(
        Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
    )
    # 检查字典是否为空
    if not first_key:
        return params_list

    params_count = len(params_dict[first_key])

    for i in params_dict.keys():
        assert params_count == len(params_dict[i]), "参数数量要彼此一致"

    for i in range(params_count):
        params = Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
        for key in params_dict.keys():
            params[key] = params_dict[key][i]
        params_list.append(params)

    return params_list


@njit(convert_params_list_dict_signature, cache=enable_cache)
def convert_params_list_dict(params_list):
    """
    将一个参数列表转换为参数字典。
    参数列表: [
      {'key1': value1, 'key2': value2},
      {'key1': value3, 'key2': value4},
    ]
    参数字典: {
      'key1': [value1, value3],
      'key2': [value2, value4],
    }
    """
    params_dict = Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])

    if len(params_list) == 0:
        # 返回一个空的字典
        return params_dict

    # 获取所有键
    first_dict = params_list[0]
    keys = List.empty_list(types.unicode_type)
    for k in first_dict.keys():
        keys.append(k)

    # 初始化输出字典，并预分配数组空间
    list_length = len(params_list)

    for key in keys:
        # Numba 不支持直接从列表中创建 NumPy 数组，需要手动填充
        array = np.empty(list_length, dtype=nb_float)
        for i in range(list_length):
            array[i] = params_list[i][key]
        params_dict[key] = array

    return params_dict
