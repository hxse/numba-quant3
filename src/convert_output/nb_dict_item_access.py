from numba import njit
from numba.typed import Dict, List
from numba.core import types
from numba.extending import overload

from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


def get_item_from_dict_list(data_list, num):
    pass


@overload(get_item_from_dict_list, jit_options={"cache": enable_cache})
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
