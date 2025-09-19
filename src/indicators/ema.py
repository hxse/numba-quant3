import numpy as np
from numba import njit
from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]


@njit(nb_float[:](nb_float[:], nb_int), cache=enable_cache)
def calc_ema(close, period):
    # 如果周期大于等于输入数据长度，或者周期小于等于 1，直接返回空数组或 NaN 数组
    if period >= len(close) or period <= 1:
        return np.full_like(close, np.nan, dtype=np.float64)

    ema = np.full_like(close, np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)

    # TA Lib 和 pandas-ta 默认行为：使用前 period 个数的 SMA 作为第一个 EMA 值
    sma_period = close[:period]
    if len(sma_period) < period:
        # 如果数据不够 period 长度，返回全 NaN 数组
        return np.full_like(close, np.nan, dtype=np.float64)

    # 计算 SMA 作为 EMA 的初始值
    initial_ema = np.sum(sma_period) / period
    ema[period - 1] = initial_ema

    # 循环计算后续的 EMA 值
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]

    return ema
