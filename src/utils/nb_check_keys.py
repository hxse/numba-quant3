import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict, List

# 从全局配置中获取 Numba 类型
from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_float = numba_config["nb"]["float"]
np_float = numba_config["np"]["float"]


@njit(cache=enable_cache)
def check_keys(keys, dict_):
    if len(keys) == 0:
        return True

    for i in keys:
        if i == "":
            continue
        if i not in dict_:
            return False

    return True


@njit(cache=enable_cache)
def check_mapping(data_mapping, ohlcv_mtf):
    for i in range(1, len(ohlcv_mtf)):
        key = f"mtf_{i}"
        if key not in data_mapping:
            return False

        if len(data_mapping[key]) != len(ohlcv_mtf[0]["time"]):
            return False

    if "skip" not in data_mapping:
        return False

    if len(data_mapping["skip"]) != len(ohlcv_mtf[0]["time"]):
        return False

    return True


@njit(cache=enable_cache)
def check_ohlcv_keys(tohlcv):
    for i in ("time", "open", "high", "low", "close", "volume"):
        if i not in tohlcv:
            return False
    return True


@njit(cache=enable_cache)
def check_ohlcv_mtf(ohlcv_mtf):
    if len(ohlcv_mtf) < 1:
        return False

    for i in ohlcv_mtf:
        if not check_ohlcv_keys(i):
            return False
    return True


@njit(cache=enable_cache)
def check_data_for_indicators(ohlcv):
    if not check_ohlcv_keys(ohlcv):
        return False
    return True


@njit(cache=enable_cache)
def check_data_for_signal(
    ohlcv_mtf, i_output_mtf_need_keys, i_output_mtf, data_mapping
):
    if not check_ohlcv_mtf(ohlcv_mtf):
        return False

    if not len(i_output_mtf_need_keys) == len(ohlcv_mtf) == len(i_output_mtf):
        return False

    for i in range(len(ohlcv_mtf)):
        exist_key = check_keys(i_output_mtf_need_keys[i], i_output_mtf[i])
        if not exist_key:
            return False

    exist_mapping = check_mapping(data_mapping, ohlcv_mtf)
    if not exist_mapping:
        return False

    return True


@njit(cache=enable_cache)
def check_data_for_backtest(
    ohlcv_mtf, s_output_need_keys, b_params_need_keys, s_output, b_params
):
    if not check_ohlcv_mtf(ohlcv_mtf):
        return False

    # 1. 输入数据校验
    if not check_keys(s_output_need_keys, s_output):
        return False

    if not check_keys(b_params_need_keys, b_params):
        return False

    return True


@njit(cache=enable_cache)
def check_data_for_performance(
    ohlcv_mtf,
    b_params_need_keys,
    b_output_need_keys,
    b_params,
    b_output,
):
    if not check_ohlcv_mtf(ohlcv_mtf):
        return False

    if not check_keys(b_params_need_keys, b_params):
        return False

    if not check_keys(b_output_need_keys, b_output):
        return False

    return True
