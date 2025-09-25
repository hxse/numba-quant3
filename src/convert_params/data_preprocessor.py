import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict, List

from src.convert_params.nb_params_signature import (
    get_data_mapping_signature,
    get_data_mapping_mtf_signature,
    get_init_tohlcv_signature,
    get_init_tohlcv_smoothed_signature,
)

from src.utils.constants import numba_config


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]


@njit(get_data_mapping_mtf_signature, cache=enable_cache)
def get_data_mapping_mtf(ohlcv_mtf):
    _d = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_int[:],
    )
    assert len(ohlcv_mtf) > 0, "ohlcv_mtf至少要有一个元素"

    # 提取基准时间序列
    # 从字典中通过 "time" 键提取数组，而不是直接对字典进行切片
    times = ohlcv_mtf[0]["time"]
    _d["skip"] = np.ones(len(times), dtype=nb_int)

    # 如果只有一个元素，无需进行映射
    if len(ohlcv_mtf) == 1:
        return _d

    # 循环处理每个 "ohlcv_mtf" 元素
    # 从索引1开始，与索引0进行映射
    for i in range(1, len(ohlcv_mtf)):
        # 从字典中通过 "time" 键提取数组
        mtf_times = ohlcv_mtf[i]["time"]

        # 使用 np.searchsorted 进行矢量化查找
        mapping_indices = np.searchsorted(mtf_times, times, side="right") - 1

        # 使用 f"mtf_{num}" 格式的键
        key = f"mtf_{i}"
        _d[key] = mapping_indices.astype(nb_int)

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
    if np_data.shape[0] == 0 or np_data.shape[1] == 0:
        return tohlcv

    assert np_data.shape[1] >= 6, "tohlcv数据列数不足"

    if smooth_mode == "ha":  # Heikin-Ashi
        new_np_data = np_data
    else:
        return tohlcv

    assert new_np_data.shape[0] == np_data.shape[0], "只允许长度相等的平滑数据"

    tohlcv["time"] = new_np_data[:, 0]
    tohlcv["open"] = new_np_data[:, 1]
    tohlcv["high"] = new_np_data[:, 2]
    tohlcv["low"] = new_np_data[:, 3]
    tohlcv["close"] = new_np_data[:, 4]
    tohlcv["volume"] = new_np_data[:, 5]
    return tohlcv
