import numpy as np
from numba import njit
from src.utils.constants import numba_config
from src.indicators.sma import calc_sma


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]


@njit(nb_float[:](nb_float[:], nb_float[:]), cache=enable_cache)
def non_zero_range(high, low):
    diff = high - low
    epsilon = np.finfo(nb_float).eps  # 约 2.22e-16
    has_zero = np.any(diff == 0)
    if has_zero:
        diff = diff + epsilon
    return diff


@njit(nb_float[:](nb_float[:], nb_int, nb_int), cache=enable_cache)
def calc_variance(close, period, ddof):
    num_data = len(close)
    if period <= 1 or num_data < period or ddof >= period:
        return np.full(num_data, np.nan, dtype=nb_float)

    # 初始化输出
    variance = np.full(num_data, np.nan, dtype=nb_float)

    # 计算累积和与累积平方和
    cumsum = np.cumsum(close)
    cumsum_sq = np.cumsum(close**2)

    # 计算滚动窗口的 sum(x) 和 sum(x^2)
    rolling_sum = np.empty(num_data, dtype=nb_float)
    rolling_sum_sq = np.empty(num_data, dtype=nb_float)

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


@njit(nb_float[:](nb_float[:], nb_int), cache=enable_cache)
def calc_stdev(close, period):
    ddof = 0
    variance = calc_variance(close, period, ddof)
    stdev_result = np.sqrt(variance)
    return stdev_result


@njit(nb_float[:, :](nb_float[:], nb_int, nb_float), cache=enable_cache)
def calc_bbands(close, length, std):
    bbands_period = length
    bbands_std_mult = std
    num_data = len(close)

    res_bbands = np.empty((num_data, 5), dtype=nb_float)
    if bbands_period <= 1 or num_data < bbands_period:
        res_bbands[:] = np.nan
        return res_bbands

    # 步骤 1: 计算中轨
    mid_band = calc_sma(close, bbands_period)

    # 步骤 2: 计算标准差
    std_dev = calc_stdev(close, bbands_period)
    deviations = bbands_std_mult * std_dev

    # 步骤 3: 计算上下轨
    upper_band = mid_band + deviations
    lower_band = mid_band - deviations

    # 步骤 4: 计算带宽和百分比
    ulr = non_zero_range(upper_band, lower_band)

    # 带宽计算：当 mid_band 为 0 时，结果应为 inf 或 nan，这里用 nan
    bandwidth = np.full(num_data, np.nan, dtype=nb_float)
    non_zero_mid_mask = mid_band != 0
    bandwidth[non_zero_mid_mask] = (
        100 * ulr[non_zero_mid_mask] / mid_band[non_zero_mid_mask]
    )

    numerator_p = non_zero_range(close, lower_band)
    percent = numerator_p / ulr  # 让 NumPy 自行处理除法和 NaN/inf 的传播

    # 将结果整合到 NumPy 数组
    res_bbands[:, 0] = upper_band
    res_bbands[:, 1] = mid_band
    res_bbands[:, 2] = lower_band
    res_bbands[:, 3] = bandwidth
    res_bbands[:, 4] = percent

    return res_bbands
