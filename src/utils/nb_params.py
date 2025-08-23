import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict, List

# 从签名文件导入签名
from src.utils.nb_params_signature import (
    get_indicator_params_signature,
    get_backtest_params_signature,
    create_params_list_template_signature,
    create_params_dict_template_signature,
    get_params_list_value_signature,
    set_params_list_value_signature,
    get_params_dict_value_signature,
    set_params_dict_value_signature,
    convert_params_dict_list_signature,
    get_data_mapping_signature,
    get_init_tohlcv_signature,
    get_init_tohlcv_smoothed_signature,
)


# 从全局配置中获取 Numba 类型
from src.utils.constants import numba_config

cache = numba_config["cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
np_float = numba_config["np"]["float"]


@njit(get_indicator_params_signature, cache=cache)
def get_indicator_params(empty):
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    if empty:
        return params
    params["sma_enable"] = nb_float(0)
    params["sma_period"] = nb_float(14)

    params["sma2_enable"] = nb_float(0)
    params["sma2_period"] = nb_float(14)

    params["bbands_enable"] = nb_float(0)
    params["bbands_period"] = nb_float(14)
    params["bbands_std_mult"] = nb_float(2.0)
    return params


@njit(get_backtest_params_signature, cache=cache)
def get_backtest_params(empty):
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    if empty:
        return params
    params["signal_select"] = nb_float(0)
    params["atr_sl_mult"] = nb_float(2.0)
    return params


@njit(create_params_list_template_signature, cache=cache)
def create_params_list_template(params_count, empty):
    assert params_count >= 0, "参数组合数量必须大于等于0"
    indicator_params_list = List.empty_list(
        Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
    )
    backtest_params_list = List.empty_list(
        Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
    )

    for n in range(params_count):
        indicator_params_list.append(get_indicator_params(empty))
        backtest_params_list.append(get_backtest_params(empty))

    return (indicator_params_list, backtest_params_list)


@njit(create_params_dict_template_signature, cache=cache)
def create_params_dict_template(params_count, empty):
    """
    根据numba文档, 目前只支持List里面放Dict, 不支持Dict里面放List
    所以,把List转成numpy数组解决问题
    """

    assert params_count > 0, "参数组合数量必须大于0"

    indicator_params_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float[:],  # 使用 NumPy 数组类型
    )

    backtest_params_dict = Dict.empty(
        key_type=types.unicode_type, value_type=nb_float[:]
    )

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


@njit(get_params_list_value_signature, cache=cache)
def get_params_list_value(key, params_list):
    params_count = len(params_list)
    arr = np.zeros(params_count, dtype=nb_float)

    for i in range(params_count):
        arr[i] = params_list[i][key]
    return arr


@njit(set_params_list_value_signature, cache=cache)
def set_params_list_value(key, params_list, arr):
    params_count = len(params_list)
    assert params_count == len(arr), (
        f"更新数量应该和原始数量一致{params_count} {len(arr)}"
    )

    for i in range(params_count):
        if i == "":
            continue
        params_list[i][key] = arr[i]


@njit(get_params_dict_value_signature, cache=cache)
def get_params_dict_value(key: str, params_dict):
    return params_dict[key]


@njit(set_params_dict_value_signature, cache=cache)
def set_params_dict_value(key: str, params_dict, arr: np.ndarray):
    first_key = ""
    for k in params_dict.keys():
        first_key = k
        break

    if first_key:
        params_count = len(params_dict[first_key])
        assert params_count == len(arr), "更新数量应该和原始数量一致"

    params_dict[key] = arr


@njit(convert_params_dict_list_signature, cache=cache)
def convert_params_dict_list(params_dict):
    first_key = ""
    for k in params_dict.keys():
        first_key = k
        break

    # 检查字典是否为空
    if not first_key:
        return List.empty_list(
            Dict.empty(
                key_type=types.unicode_type,
                value_type=nb_float,
            )
        )

    params_count = len(params_dict[first_key])

    for i in params_dict.keys():
        assert params_count == len(params_dict[i]), "参数数量要彼此一致"

    params_list = List.empty_list(
        Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
    )

    for i in range(params_count):
        params = Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
        for key in params_dict.keys():
            params[key] = params_dict[key][i]
        params_list.append(params)

    return params_list


@njit(get_data_mapping_signature, cache=cache)
def get_data_mapping(tohlcv_np, tohlcv_np_mtf):
    _d = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_int[:],
    )
    if (
        tohlcv_np is None
        or tohlcv_np_mtf is None
        or tohlcv_np.shape[0] == 0
        or tohlcv_np_mtf.shape[0] == 0
    ):
        return _d

    times = tohlcv_np[:, 0]
    mtf_times = tohlcv_np_mtf[:, 0]

    # 核心优化：使用 np.searchsorted 进行矢量化查找
    # side='right' 找到第一个大于当前时间戳的位置
    mapping_indices = np.searchsorted(mtf_times, times, side="right") - 1

    _d["mtf"] = mapping_indices.astype(nb_int)
    return _d


@njit(get_init_tohlcv_signature, cache=cache)
def init_tohlcv(np_data):
    tohlcv = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float[:],
    )
    if np_data is None:
        return tohlcv
    assert np_data.shape[1] >= 6, "tohlcv数据列数不足"
    tohlcv["time"] = np_data[:, 0]
    tohlcv["open"] = np_data[:, 1]
    tohlcv["high"] = np_data[:, 2]
    tohlcv["low"] = np_data[:, 3]
    tohlcv["close"] = np_data[:, 4]
    tohlcv["volume"] = np_data[:, 5]
    return tohlcv


@njit(get_init_tohlcv_smoothed_signature, cache=cache)
def init_tohlcv_smoothed(np_data, smooth_mode):
    """
    允许长度相等的平滑数据如Heikin-Ashi
    不允许长度不相等的平滑数据如renko
    """
    tohlcv = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float[:],
    )
    if np_data is None:
        return tohlcv
    assert np_data.shape[1] >= 6, "tohlcv数据列数不足"

    if smooth_mode is None:
        return tohlcv
    elif smooth_mode == "":
        new_np_data = np_data
    else:
        raise KeyError(f"not match mode {smooth_mode}")

    assert new_np_data.shape[0] == np_data.shape[0], "只允许长度相等的平滑数据"

    tohlcv["time"] = new_np_data[:, 0]
    tohlcv["open"] = new_np_data[:, 1]
    tohlcv["high"] = new_np_data[:, 2]
    tohlcv["low"] = new_np_data[:, 3]
    tohlcv["close"] = new_np_data[:, 4]
    tohlcv["volume"] = new_np_data[:, 5]
    return tohlcv
