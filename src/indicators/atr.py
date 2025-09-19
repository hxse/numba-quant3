import numpy as np
from numba import njit
from src.utils.constants import numba_config


from src.indicators.rsi import calc_rma


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]


@njit(nb_float[:](nb_float[:], nb_float[:], nb_float[:]), cache=enable_cache)
def calc_tr(high, low, close):
    """
    矢量化计算真实波幅 (True Range, TR) 的函数。
    """
    n = high.size
    if n == 0:
        return np.full(0, np.nan, dtype=nb_float)

    # 1. 矢量化计算三个波幅值
    h_l = high - low
    h_pc = np.full(n, np.nan, dtype=nb_float)
    l_pc = np.full(n, np.nan, dtype=nb_float)

    h_pc[1:] = np.abs(high[1:] - close[:-1])
    l_pc[1:] = np.abs(low[1:] - close[:-1])

    # 2. 矢量化计算 True Range
    tr = np.maximum(h_l, np.maximum(h_pc, l_pc))
    return tr


@njit(nb_float[:](nb_float[:], nb_float[:], nb_float[:], nb_int), cache=enable_cache)
def calc_atr(high, low, close, period):
    """
    ATR计算函数，与TA-Lib逻辑完全一致，并进行了矢量化优化。
    """
    n = high.size
    if n < period + 1:
        return np.full_like(high, np.nan, dtype=nb_float)

    # 1. 调用新的 calc_tr 函数来计算 True Range
    tr = calc_tr(high, low, close)

    # 2. 对 True Range 进行 RMA 平滑处理
    # TA-Lib的ATR计算基于TR[1:]，所以我们只对TR的有效部分进行计算
    valid_tr = tr[1:]

    if valid_tr.size < period:
        return np.full_like(high, np.nan, dtype=nb_float)

    atr_values = calc_rma(valid_tr, period)

    # 3. 结果对齐
    result = np.full_like(high, np.nan, dtype=nb_float)
    result[period:] = atr_values[period - 1 :]

    return result
