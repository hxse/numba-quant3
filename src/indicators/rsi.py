import numpy as np
from numba import njit, float64
from src.utils.constants import numba_config

cache = numba_config["cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]


@njit(nb_float[:](nb_float[:], nb_int), cache=cache)
def nb_rma_optimized(series, length):
    """
    一个更接近TA-Lib逻辑的RMA实现。
    """
    n = series.size
    result = np.full(n, np.nan, dtype=nb_float)

    # RMA 的第一个有效值需要 length 个数据点
    if n < length:
        return result

    # 计算前 length 个数据的平均值作为第一个 RMA 值
    current_sum = 0.0
    for i in range(length):
        current_sum += series[i]

    result[length - 1] = current_sum / length

    alpha = 1.0 / length
    for i in range(length, n):
        result[i] = (series[i] - result[i - 1]) * alpha + result[i - 1]

    return result


@njit(nb_float[:](nb_float[:], nb_int), cache=cache)
def calc_rsi(close, length=14):
    """
    计算相对强弱指数（RSI），其逻辑与TA-Lib更接近。
    """
    n = close.size
    if n < length:
        return np.full(n, np.nan, dtype=nb_float)

    # 手动计算价格变化，避免使用 np.diff
    diffs = np.full(n, np.nan, dtype=float64)
    for i in range(1, n):
        diffs[i] = close[i] - close[i - 1]

    # 分离上涨和下跌部分
    ups = np.where(diffs > 0, diffs, 0.0)
    downs = np.where(diffs < 0, np.abs(diffs), 0.0)

    # 调整ups和downs数组的起始点，使其与TA-Lib的逻辑匹配
    # diffs的第一个元素是NaN，因此ups和downs的第一个元素也应被忽略
    ups_adjusted = ups[1:]
    downs_adjusted = downs[1:]

    # 计算上涨和下跌的平均值（使用 RMA）
    # nb_rma_optimized函数将处理其自身的NaN
    avg_ups = nb_rma_optimized(ups_adjusted, length)
    avg_downs = nb_rma_optimized(downs_adjusted, length)

    # 计算相对强弱（RS），手动处理除零
    rs = np.full(n - 1, np.nan, dtype=nb_float)
    for i in range(n - 1):
        if avg_downs[i] != 0:
            rs[i] = avg_ups[i] / avg_downs[i]

    # 计算 RSI
    rsi_result = 100.0 - (100.0 / (1.0 + rs))

    # 在RSI结果前填充length个NaN，以匹配TA-Lib的输出
    result = np.full(n, np.nan, dtype=nb_float)
    result[length:] = rsi_result[length - 1 :]

    return result
