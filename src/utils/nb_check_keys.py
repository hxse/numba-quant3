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
    elif len(keys) == 1 and keys[0] == "":
        return True

    for i in keys:
        if i not in dict_:
            return False

    return True


@njit(cache=cache)
def check_mapping(signal_1_keys_large, mapping_large, data_count):
    if len(signal_1_keys_large) > 0:
        if mapping_large is None:
            return False
        if len(mapping_large) == 0:
            return False
        if len(mapping_large) != data_count:
            return False

    return True


@njit(cache=cache)
def check_all(
    data_count,
    signal_1_keys,
    signal_1_keys_large,
    indicator_output,
    indicators_output_large,
    mapping_large,
):
    exist_key = check_keys(signal_1_keys, indicator_output)
    if not exist_key:
        return False

    exist_key = check_keys(signal_1_keys_large, indicators_output_large)
    if not exist_key:
        return False

    exist_mapping = check_mapping(signal_1_keys_large, mapping_large, data_count)
    if not exist_mapping:
        return False

    return True
