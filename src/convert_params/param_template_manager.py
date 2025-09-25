import numpy as np
import numba as nb
from numba import njit
from numba.typed import Dict, List
from numba.core.types import unicode_type

from src.convert_params.param_template import get_backtest_params, get_indicator_params
from src.convert_params.nb_params_signature import (
    create_indicator_params_list_signature,
    create_backtest_params_list_signature,
    create_params_dict_template_signature,
    get_params_list_value_signature,
    set_params_list_value_signature,
    set_params_list_value_mtf_signature,
    get_params_dict_value_signature,
    set_params_dict_value_signature,
)


from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


@njit(create_indicator_params_list_signature, cache=enable_cache)
def create_indicator_params_list(params_count, mtf_count, empty):
    """
    创建并返回一个嵌套列表，
    外层由 params_count 控制，内层由 mtf_count 控制。
    """

    # 定义内部字典列表的类型
    inner_list_type = List.empty_list(
        Dict.empty(
            key_type=unicode_type,
            value_type=nb_float,
        )
    )

    # 创建一个空的 Numba 列表，用于存放所有内部列表
    indicator_params_list_final = List.empty_list(inner_list_type)

    # 外层循环：根据参数组合数量 (params_count)
    for _ in range(params_count):
        # 为每个参数组合创建一个新的空列表，用于存放 mtf 数据
        mtf_params_list = List.empty_list(
            Dict.empty(
                key_type=unicode_type,
                value_type=nb_float,
            )
        )

        # 内层循环：根据时间框架数量 (mtf_count)
        for _ in range(mtf_count):
            # 向当前时间框架列表中添加一个指标参数字典
            mtf_params_list.append(get_indicator_params(empty))

        # 将已填充的、代表一个参数组合的 mtf 列表添加到最终列表中
        indicator_params_list_final.append(mtf_params_list)

    return indicator_params_list_final


@njit(create_backtest_params_list_signature, cache=enable_cache)
def create_backtest_params_list(params_count, empty):
    """
    创建并返回回测参数列表。
    """
    assert params_count >= 0, "参数组合数量必须大于等于0"

    backtest_params_list = List.empty_list(
        Dict.empty(
            key_type=unicode_type,
            value_type=nb_float,
        )
    )

    for _ in range(params_count):
        backtest_params_list.append(get_backtest_params(empty))

    return backtest_params_list


@njit(create_params_dict_template_signature, cache=enable_cache)
def create_params_dict_template(params_count, empty):
    """
    根据numba文档, 目前只支持List里面放Dict, 不支持Dict里面放List
    所以,把List转成numpy数组解决问题
    """

    assert params_count > 0, "参数组合数量必须大于0"

    indicator_params_dict = Dict.empty(
        key_type=unicode_type,
        value_type=nb_float[:],  # 使用 NumPy 数组类型
    )

    backtest_params_dict = Dict.empty(key_type=unicode_type, value_type=nb_float[:])

    params = get_indicator_params(empty)
    for key in params.keys():
        # 直接创建 NumPy 数组，填充相同的值
        arr = np.zeros(params_count, dtype=nb_float)
        arr[:] = params[key]
        indicator_params_dict[key] = arr

    params = get_backtest_params(empty)
    for key in params.keys():
        # 直接创建 NumPy 数组，填充相同的值
        arr = np.zeros(params_count, dtype=nb_float)
        arr[:] = params[key]
        backtest_params_dict[key] = arr

    return (indicator_params_dict, backtest_params_dict)


@njit(get_params_list_value_signature, cache=enable_cache)
def get_params_list_value(key, params_list):
    params_count = len(params_list)
    arr = np.zeros(params_count, dtype=nb_float)

    for i in range(params_count):
        arr[i] = params_list[i][key]
    return arr


@njit(set_params_list_value_signature, cache=enable_cache)
def set_params_list_value(key, params_list, arr):
    params_count = len(params_list)
    assert params_count == len(arr), (
        f"更新数量应该和原始数量一致{params_count} {len(arr)}"
    )
    if key == "":
        return

    for i in range(params_count):
        params_list[i][key] = arr[i]


@njit(set_params_list_value_mtf_signature, cache=enable_cache)
def set_params_list_value_mtf(mtf_idx, key, params_list, arr):
    """
    更新嵌套列表中的指标参数值，并对每个元素的索引进行越界检测。
    """
    params_count = len(params_list)

    # 对 params_list 和 arr 的长度进行越界检测
    assert params_count == len(arr), (
        f"更新数量应该和原始数量一致: {params_count} != {len(arr)}"
    )

    if key == "":
        return

    for i in range(params_count):
        # 逐个检查 mtf_idx 是否在当前内部列表的有效范围内
        assert 0 <= mtf_idx < len(params_list[i]), (
            f"mtf_idx 越界: {mtf_idx} 超出索引 {i} 的有效范围 [0, {len(params_list[i]) - 1}]"
        )

        params_list[i][mtf_idx][key] = arr[i]


@njit(get_params_dict_value_signature, cache=enable_cache)
def get_params_dict_value(key: str, params_dict):
    return params_dict[key]


@njit(set_params_dict_value_signature, cache=enable_cache)
def set_params_dict_value(key: str, params_dict, arr: np.ndarray):
    first_key = ""
    for k in params_dict.keys():
        first_key = k
        break

    if first_key:
        params_count = len(params_dict[first_key])
        assert params_count == len(arr), "更新数量应该和原始数量一致"

    params_dict[key] = arr
