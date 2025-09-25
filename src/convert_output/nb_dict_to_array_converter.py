from numba import njit, prange, typeof
from numba.typed import Dict, List
from numba.core import types
import numpy as np
from numba.extending import overload

from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


@njit(cache=enable_cache)
def merge_dict_to_np_array_wrapper(params_dict):
    return merge_dict_to_np_array(params_dict)


def merge_dict_to_np_array(params_dict):
    pass


@overload(merge_dict_to_np_array, jit_options={"cache": enable_cache})
def ov_merge_dict_to_np_array(params_dict):
    # 将重复的检查提取到最前面
    if not (
        isinstance(params_dict, types.DictType)
        # and params_dict.key_type == types.unicode_type
    ):
        return None

    # 定义核心实现逻辑 (用于数组值)
    @njit(parallel=True, cache=enable_cache)
    def convert_impl_array(params_dict, dtype):
        keys = get_dict_keys_wrapper(params_dict)

        num_keys = len(keys)

        if num_keys == 0:
            return keys, np.empty((0, 0), dtype=dtype)

        first_value_array_len = len(params_dict[keys[0]])

        shape = (first_value_array_len, num_keys)
        result_array = np.empty(shape, dtype=dtype)

        # 验证每个数组的长度是否一致
        for key in keys:
            assert len(params_dict[key]) == first_value_array_len

        for i in prange(num_keys):
            _i = nb_int(i)
            key = keys[_i]
            result_array[:, _i] = params_dict[key]

        return keys, result_array

    # 定义核心实现逻辑 (用于标量值)
    @njit(cache=enable_cache, parallel=True)
    def convert_impl_scalar(params_dict, dtype):
        keys = get_dict_keys_wrapper(params_dict)

        num_keys = len(keys)

        if num_keys == 0:
            return keys, np.empty((0), dtype=dtype)

        shape = (num_keys,)
        result_array = np.empty(shape, dtype=dtype)

        for i in prange(num_keys):
            _i = nb_int(i)
            key = keys[_i]
            result_array[_i] = params_dict[key]

        return keys, result_array

    # 使用字典来映射类型到实现函数和dtype
    dispatch_map = {
        nb_int[:]: (convert_impl_array, nb_int),
        nb_float[:]: (convert_impl_array, nb_float),
        nb_bool[:]: (convert_impl_array, nb_bool),
        nb_int: (convert_impl_scalar, nb_int),
        nb_float: (convert_impl_scalar, nb_float),
        nb_bool: (convert_impl_scalar, nb_bool),
    }

    # 检查字典的值类型并返回相应的实现
    value_type = params_dict.value_type

    # 查找匹配的实现函数和dtype
    if value_type in dispatch_map:
        impl_func, dtype = dispatch_map[value_type]

        # 返回闭包函数
        def impl(params_dict):
            return impl_func(params_dict, dtype)

        return impl

    return None


# 这是我们要重载的 Python 函数
@njit(cache=enable_cache)
def get_dict_keys_and_values_wrapper(params_dict):
    return get_dict_keys_and_values(params_dict)


# 这是我们要重载的 Python 函数
def get_dict_keys_and_values(params_dict):
    """一个占位符函数，用于 Numba 重载"""
    pass


# 这是我们重载的实现
@overload(get_dict_keys_and_values, jit_options={"cache": enable_cache})
def ov_get_dict_keys_and_values(params_dict):
    """
    Numba 重载函数，利用闭包动态生成实现。
    """
    # 1. 检查输入是否为 Numba Dict 类型对象
    assert isinstance(params_dict, types.DictType), (
        f"需要字典类型, 实际为{type(params_dict)}"
    )

    # 2. 从类型对象中获取键和值类型
    key_type = params_dict.key_type
    value_type = params_dict.value_type

    # 4. 返回一个纯 Python 函数作为实现
    # 确保参数名称与 ov_get_dict_keys_and_values 一致
    def impl(params_dict):
        # 这里的 params_dict 是一个实际的 Numba Dict 实例
        keys = List.empty_list(key_type)
        values = List.empty_list(value_type)

        for key, value in params_dict.items():
            keys.append(key)
            values.append(value)

        return (keys, values)

    # 5. 返回 impl 函数。
    return impl


# 这是我们要重载的 Python 函数
@njit(cache=enable_cache)
def get_dict_keys_wrapper(params_dict):
    return get_dict_keys(params_dict)


# 这是我们要重载的 Python 函数
def get_dict_keys(params_dict):
    """一个占位符函数，用于 Numba 重载"""
    pass


# 这是我们重载的实现
@overload(get_dict_keys, jit_options={"cache": enable_cache})
def ov_get_dict_keys(params_dict):
    """
    Numba 重载函数，利用闭包动态生成实现。
    """
    # import pdb

    # pdb.set_trace()
    assert isinstance(params_dict, types.DictType), f"需要字典类型, 实际为{params_dict}"

    # 2. 从类型对象中获取键和值类型
    key_type = params_dict.key_type

    # 4. 返回一个纯 Python 函数作为实现
    # 确保参数名称与 ov_get_dict_keys_and_values 一致
    def impl(params_dict):
        # 这里的 params_dict 是一个实际的 Numba Dict 实例
        keys = List.empty_list(key_type)

        for key, value in params_dict.items():
            keys.append(key)

        return keys

    # 5. 返回 impl 函数。
    return impl


# 这是我们要重载的 Python 函数
@njit(cache=enable_cache)
def get_dict_values_wrapper(params_dict):
    return get_dict_values(params_dict)


# 这是我们要重载的 Python 函数
def get_dict_values(params_dict):
    """一个占位符函数，用于 Numba 重载"""
    pass


# 这是我们重载的实现
@overload(get_dict_values, jit_options={"cache": enable_cache})
def ov_get_dict_values(params_dict):
    """
    Numba 重载函数，利用闭包动态生成实现。
    """
    # 1. 检查输入是否为 Numba Dict 类型对象
    assert isinstance(params_dict, types.DictType), (
        f"需要字典类型, 实际为{type(params_dict)}"
    )

    value_type = params_dict.value_type

    # 4. 返回一个纯 Python 函数作为实现
    # 确保参数名称与 ov_get_dict_keys_and_values 一致
    def impl(params_dict):
        # 这里的 params_dict 是一个实际的 Numba Dict 实例
        values = List.empty_list(value_type)

        for key, value in params_dict.items():
            values.append(value)

        return values

    # 5. 返回 impl 函数。
    return impl
