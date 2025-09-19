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

print("params cache", enable_cache)


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
def check_mapping(signal_keys_mtf, data_mapping, data_count):
    # 使用一个静态元组来存储需要检查的键
    # 元组在 Numba 中是类型确定的，可以安全地遍历
    for key in ("mtf", "skip"):
        # 处理 mtf 的特殊条件：如果 signal_keys_mtf 长度为0，则跳过对 mtf 的检查
        if key == "mtf" and len(signal_keys_mtf) == 0:
            continue

        # 对当前的 key 执行通用的检查
        if key not in data_mapping:
            return False

        _item = data_mapping[key]
        if len(_item) == 0:
            return False

        if len(_item) != data_count:
            return False

    return True


@njit(cache=enable_cache)
def check_tohlcv_keys(tohlcv):
    for i in ("time", "open", "high", "low", "close", "volume"):
        if i not in tohlcv:
            return False
    return True


@njit(cache=enable_cache)
def check_all(
    _tohlcv,
    _tohlcv_mtf,
    signal_keys,
    signal_keys_mtf,
    indicator_output,
    indicators_output_mtf,
    data_mapping,
):
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

    exist_mapping = check_mapping(signal_keys_mtf, data_mapping, data_count)
    if not exist_mapping:
        return False

    return True
