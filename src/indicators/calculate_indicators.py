import numpy as np
from numba import njit
from src.utils.constants import numba_config

from src.utils.nb_check_keys import check_data_for_indicators

from src.parallel_signature import indicators_signature


from .sma import calc_sma
from .ema import calc_ema
from .bbands import calc_bbands
from .rsi import calc_rsi
from .atr import calc_atr
from .psar import calc_psar

from enum import IntEnum


enable_cache = numba_config["enable_cache"]
nb_float = numba_config["nb"]["float"]


class MaxIndicatorCount(IntEnum):
    sma = 3
    ema = 3
    bbands = 1
    rsi = 1
    atr = 1
    psar = 1


mic = MaxIndicatorCount


@njit(indicators_signature, cache=enable_cache)
def calc_indicators(ohlcv, i_params, i_output):
    if not check_data_for_indicators(ohlcv):
        return

    for i in range(mic.sma.value):
        if f"sma_enable_{i}" in i_params and i_params[f"sma_enable_{i}"]:
            i_output[f"sma_{i}"] = calc_sma(ohlcv["close"], i_params[f"sma_period_{i}"])

    for i in range(mic.ema.value):
        if f"ema_enable_{i}" in i_params and i_params[f"ema_enable_{i}"]:
            i_output[f"ema_{i}"] = calc_ema(ohlcv["close"], i_params[f"ema_period_{i}"])

    for i in range(mic.bbands.value):
        if f"bbands_enable_{i}" in i_params and i_params[f"bbands_enable_{i}"]:
            bbands = calc_bbands(
                ohlcv["close"],
                int(i_params[f"bbands_period_{i}"]),
                i_params[f"bbands_std_mult_{i}"],
            )
            i_output[f"bbands_upper_{i}"] = bbands[:, 0]
            i_output[f"bbands_middle_{i}"] = bbands[:, 1]
            i_output[f"bbands_lower_{i}"] = bbands[:, 2]
            i_output[f"bbands_bandwidth_{i}"] = bbands[:, 3]
            i_output[f"bbands_percent_{i}"] = bbands[:, 4]

    for i in range(mic.rsi.value):
        if f"rsi_enable_{i}" in i_params and i_params[f"rsi_enable_{i}"]:
            i_output[f"rsi_{i}"] = calc_rsi(ohlcv["close"], i_params[f"rsi_period_{i}"])

    for i in range(mic.atr.value):
        if f"atr_enable_{i}" in i_params and i_params[f"atr_enable_{i}"]:
            i_output[f"atr_{i}"] = calc_atr(
                ohlcv["high"],
                ohlcv["low"],
                ohlcv["close"],
                i_params[f"atr_period_{i}"],
            )

    for i in range(mic.psar.value):
        if f"psar_enable_{i}" in i_params and i_params[f"psar_enable_{i}"]:
            psar = calc_psar(
                ohlcv["high"],
                ohlcv["low"],
                ohlcv["close"],
                i_params[f"psar_af0_{i}"],
                i_params[f"psar_af_step_{i}"],
                i_params[f"psar_max_af_{i}"],
            )
            i_output[f"psar_long_{i}"] = psar[:, 0]
            i_output[f"psar_short_{i}"] = psar[:, 1]
            i_output[f"psar_af_{i}"] = psar[:, 2]
            i_output[f"psar_reversal_{i}"] = psar[:, 3]
