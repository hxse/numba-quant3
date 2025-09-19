import numpy as np
from numba import njit
from src.utils.constants import numba_config


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]


@njit(nb_float[:](nb_float[:], nb_int), cache=enable_cache)
def calc_sma(close, period):
    num_data = len(close)
    if period <= 1 or num_data < period:
        return np.full((num_data,), np.nan, dtype=nb_float)

    # 使用 np.ones() 创建一个权重数组
    weights = np.ones(period, dtype=nb_float) / period

    # 使用 np.convolve 进行卷积计算
    # 'valid' 模式只返回完全重叠的部分，长度为 len(close) - period + 1
    # 结果数组的长度为 100 - 14 + 1 = 87
    sma_result = np.convolve(close, weights, mode="valid")

    # 在前面填充 NaN
    # 填充 period - 1 个 NaN，长度为 13
    # 13 + 87 = 100，长度匹配！
    nan_prefix = np.full((period - 1,), np.nan, dtype=nb_float)
    return np.concatenate((nan_prefix, sma_result))
