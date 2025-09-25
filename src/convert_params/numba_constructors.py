import numpy as np
import numba as nb
from numba import njit, typeof
from numba.core import types
from numba.core.types import unicode_type
from numba.typed import Dict, List


# 从全局配置中获取 Numba 类型
from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_float = numba_config["nb"]["float"]
np_float = numba_config["np"]["float"]


@njit(cache=enable_cache)
def create_list_unicode_empty():
    _list = List.empty_list(types.unicode_type)
    return _list


@njit(cache=enable_cache)
def create_list_unicode_one():
    _list = List.empty_list(types.unicode_type)
    _list.append("")
    return _list


@njit(cache=enable_cache)
def create_list_dict_float_1d_empty():
    _dict = Dict.empty(key_type=unicode_type, value_type=nb_float[:])
    _list = List.empty_list(_dict)
    return _list


@njit(cache=enable_cache)
def create_list_dict_float_1d_one():
    _dict = Dict.empty(key_type=unicode_type, value_type=nb_float[:])
    _list = List.empty_list(_dict)

    _d = Dict.empty(key_type=unicode_type, value_type=nb_float[:])
    _d["time"] = np.array([1.0], dtype=nb_float)
    _list.append(_d)
    return _list


@njit(cache=enable_cache)
def create_dict_float_1d_empty():
    return Dict.empty(key_type=unicode_type, value_type=nb_float[:])


@njit(cache=enable_cache)
def create_dict_float_1d_one():
    _ = Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
    _["time"] = np.array([1.0], dtype=nb_float)
    return _


@njit(cache=enable_cache)
def create_2d_list_unicode_empty():
    outer_list = List.empty_list(List.empty_list(types.unicode_type))

    inner_list_1 = List.empty_list(types.unicode_type)
    outer_list.append(inner_list_1)

    inner_list_2 = List.empty_list(types.unicode_type)
    outer_list.append(inner_list_2)

    return outer_list


@njit(cache=enable_cache)
def create_2d_list_unicode_one():
    outer_list = List.empty_list(List.empty_list(types.unicode_type))

    inner_list_1 = List.empty_list(types.unicode_type)
    inner_list_1.append("")
    outer_list.append(inner_list_1)

    inner_list_2 = List.empty_list(types.unicode_type)
    inner_list_2.append("")
    outer_list.append(inner_list_2)

    return outer_list
