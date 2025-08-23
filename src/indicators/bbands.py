import numpy as np
from numba import njit, float64, int64
from src.utils.constants import numba_config
from src.indicators.sma import calc_sma

cache = numba_config["cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]


# Numba 兼容的滚动方差函数（完全矢量化，模仿 pandas-ta 的 variance）
@njit(float64[:](float64[:], int64, int64), cache=True)
def calc_variance(close, period, ddof):
    num_data = len(close)
    if period <= 1 or num_data < period or ddof >= period:
        return np.full(num_data, np.nan, dtype=float64)

    # 初始化输出
    variance = np.full(num_data, np.nan, dtype=float64)

    # 计算累积和与累积平方和
    cumsum = np.cumsum(close)
    cumsum_sq = np.cumsum(close**2)

    # 计算滚动窗口的 sum(x) 和 sum(x^2)
    rolling_sum = np.empty(num_data, dtype=float64)
    rolling_sum_sq = np.empty(num_data, dtype=float64)

    # 前 period-1 个值为 NaN
    rolling_sum[: period - 1] = np.nan
    rolling_sum_sq[: period - 1] = np.nan

    # 第一个有效窗口
    rolling_sum[period - 1] = cumsum[period - 1]
    rolling_sum_sq[period - 1] = cumsum_sq[period - 1]

    # 后续窗口
    if num_data > period:
        rolling_sum[period:] = cumsum[period:] - cumsum[:-period]
        rolling_sum_sq[period:] = cumsum_sq[period:] - cumsum_sq[:-period]

    # 计算方差：var = [sum(x^2) - (sum(x)^2 / n)] / (n - ddof)
    n = period
    denominator = n - ddof
    if denominator <= 0:
        return variance
    variance[period - 1 :] = (
        rolling_sum_sq[period - 1 :] - (rolling_sum[period - 1 :] ** 2 / n)
    ) / denominator

    return variance


# Numba 兼容的标准差函数（模仿 pandas-ta 的 stdev）
# --- 唯一的修改在这里 ---
@njit(nb_float[:](nb_float[:], nb_int), cache=cache)
def calc_stdev(close, period):
    # 将 ddof 从 1 改为 0，以匹配 pandas-ta 中 bbands 的实际行为
    ddof = 0
    variance = calc_variance(close, period, ddof)
    stdev_result = np.sqrt(variance)
    return stdev_result


# bbands 函数完全不需要修改
@njit(nb_float[:, :](nb_float[:], nb_int, nb_float), cache=cache)
def calc_bbands(close, length, std):
    bbands_period = length
    bbands_std_mult = std
    num_data = len(close)

    res_bbands = np.empty((num_data, 5), dtype=nb_float)
    if bbands_period <= 1 or num_data < bbands_period:
        res_bbands[:] = np.nan
        return res_bbands

    mid_band = calc_sma(close, bbands_period)
    std_dev = calc_stdev(close, bbands_period)
    deviations = bbands_std_mult * std_dev
    upper_band = mid_band + deviations
    lower_band = mid_band - deviations
    ulr = upper_band - lower_band
    bandwidth = np.where(mid_band != 0, 100 * ulr / mid_band, np.nan)
    percent = np.where(ulr != 0, (close - lower_band) / ulr, np.nan)

    res_bbands[:, 0] = upper_band
    res_bbands[:, 1] = mid_band
    res_bbands[:, 2] = lower_band
    res_bbands[:, 3] = bandwidth
    res_bbands[:, 4] = percent

    return res_bbands
