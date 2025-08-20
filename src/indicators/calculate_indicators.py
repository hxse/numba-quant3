import numpy as np
from numba import njit
from src.utils.constants import numba_config


from .sma import calc_sma
from .bbands import calc_bbands

cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]


@njit(cache=cache)
def calc_indicators(indicator_output, close, param):
    if param["sma_enable"]:
        indicator_output["sma"] = calc_sma(close, param["sma_period"])

    if param["sma2_enable"]:
        indicator_output["sma2"] = calc_sma(close, param["sma2_period"])

    if param["bbands_enable"]:
        bbands = calc_bbands(close, param["bbands_period"], param["bbands_std_mult"])
        indicator_output["bbands_upper"] = bbands[:, 0]
        indicator_output["bbands_middle"] = bbands[:, 1]
        indicator_output["bbands_lower"] = bbands[:, 2]
