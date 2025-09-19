from numba import njit, prange
from numba.typed import Dict, List
from numba.core import types
import numpy as np
from numba.extending import overload
from src.convert_output.nb_dict_utils import get_dict_keys_as_list

from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


def convert_dict_to_np_array(params_dict):
    pass


@overload(convert_dict_to_np_array, jit_options={"cache": enable_cache})
def ov_convert_dict_to_2d_array(params_dict):
    # 将重复的检查提取到最前面
    if not (
        isinstance(params_dict, types.DictType)
        and params_dict.key_type == types.unicode_type
    ):
        return None

    # 定义核心实现逻辑 (用于数组值)
    @njit(parallel=True, cache=enable_cache)
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
    @njit(parallel=True, cache=enable_cache)
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
