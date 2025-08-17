import numpy as np
from numba import njit
from src.utils.constants import numba_config


cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]


@njit(cache=cache)
def calc_sma(close, sma_period):
    num_data = len(close)
    res_sma = np.empty(num_data, dtype=nb_float)
    res_sma = close + sma_period
    return res_sma
