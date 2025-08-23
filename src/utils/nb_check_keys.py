import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict, List

# 从全局配置中获取 Numba 类型
from src.utils.constants import numba_config

cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]
np_float = numba_config["np"]["float"]

print("params cache", cache)


@njit(cache=cache)
def check_keys(keys, dict_):
    if len(keys) == 0:
        return True

    for i in keys:
        if i == "":
            continue
        if i not in dict_:
            return False

    return True


@njit(cache=cache)
def check_mapping(signal_keys_mtf, mapping_mtf, data_count):
    if len(signal_keys_mtf) > 0:
        if "mtf" not in mapping_mtf:
            return False
        m_mtf = mapping_mtf["mtf"]
        if len(m_mtf) == 0:
            return False
        if len(m_mtf) != data_count:
            return False

    return True


@njit(cache=cache)
def check_tohlcv_keys(tohlcv):
    for i in ("time", "open", "high", "low", "close", "volume"):
        if i not in tohlcv:
            return False
    return True


@njit(cache=cache)
def check_all(
    _tohlcv,
    _tohlcv_mtf,
    signal_keys,
    signal_keys_mtf,
    indicator_output,
    indicators_output_mtf,
    mapping_mtf,
):
    if len(signal_keys) > -1:
        if not check_tohlcv_keys(_tohlcv):
            return False

    if len(signal_keys_mtf) > 0:
        if not check_tohlcv_keys(_tohlcv_mtf):
            return False

    data_count = len(_tohlcv["close"])

    exist_key = check_keys(signal_keys, indicator_output)
    if not exist_key:
        return False

    exist_key = check_keys(signal_keys_mtf, indicators_output_mtf)
    if not exist_key:
        return False

    exist_mapping = check_mapping(signal_keys_mtf, mapping_mtf, data_count)
    if not exist_mapping:
        return False

    return True
