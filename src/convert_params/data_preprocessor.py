import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict, List

from src.convert_params.nb_params_signature import (
    get_data_mapping_signature,
    get_init_tohlcv_signature,
    get_init_tohlcv_smoothed_signature,
)

from src.utils.constants import numba_config


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]


@njit(get_data_mapping_signature, cache=enable_cache)
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
    _d["skip"] = np.ones(len(times), dtype=nb_int)  # 1 是不跳过，0 是跳过，默认值为1
    return _d


@njit(get_init_tohlcv_signature, cache=enable_cache)
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


@njit(get_init_tohlcv_smoothed_signature, cache=enable_cache)
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
