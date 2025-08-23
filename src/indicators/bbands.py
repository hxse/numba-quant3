import numpy as np
from numba import njit
from src.utils.constants import numba_config

cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]


@njit(cache=cache)
def calc_bbands(tohlcv, bbands_period, bbands_std_mult):
    close = tohlcv["close"]
    num_data = len(close)
    res_bbands = np.empty((num_data, 3), dtype=nb_float)
    res_bbands[:, 0] = close + bbands_period + 1 * bbands_std_mult
    res_bbands[:, 1] = close + bbands_period + 2 * bbands_std_mult
    res_bbands[:, 2] = close + bbands_period + 3 * bbands_std_mult

    return res_bbands
