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
