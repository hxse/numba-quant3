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

enable_cache = numba_config["enable_cache"]
nb_float = numba_config["nb"]["float"]


@njit(indicators_signature, cache=enable_cache)
def calc_indicators(ohlcv, i_params, i_output):
    if not check_data_for_indicators(ohlcv):
        return

    for i in ["sma", "sma2"]:
        if f"{i}_enable" in i_params and i_params[f"{i}_enable"]:
            i_output[i] = calc_sma(ohlcv["close"], i_params[f"{i}_period"])
    for i in ["ema", "ema2"]:
        if f"{i}_enable" in i_params and i_params[f"{i}_enable"]:
            i_output[i] = calc_ema(ohlcv["close"], i_params[f"{i}_period"])

    if "bbands_enable" in i_params and i_params["bbands_enable"]:
        bbands = calc_bbands(
            ohlcv["close"],
            int(i_params["bbands_period"]),
            i_params["bbands_std_mult"],
        )
        i_output["bbands_upper"] = bbands[:, 0]
        i_output["bbands_middle"] = bbands[:, 1]
        i_output["bbands_lower"] = bbands[:, 2]
        i_output["bbands_bandwidth"] = bbands[:, 3]
        i_output["bbands_percent"] = bbands[:, 4]

    if "rsi_enable" in i_params and i_params["rsi_enable"]:
        i_output["rsi"] = calc_rsi(ohlcv["close"], i_params["rsi_period"])

    if "atr_enable" in i_params and i_params["atr_enable"]:
        i_output["atr"] = calc_atr(
            ohlcv["high"],
            ohlcv["low"],
            ohlcv["close"],
            i_params["atr_period"],
        )

    if "psar_enable" in i_params and i_params["psar_enable"]:
        psar = calc_psar(
            ohlcv["high"],
            ohlcv["low"],
            ohlcv["close"],
            i_params["psar_af0"],
            i_params["psar_af_step"],
            i_params["psar_max_af"],
        )
        i_output["psar_long"] = psar[:, 0]
        i_output["psar_short"] = psar[:, 1]
        i_output["psar_af"] = psar[:, 2]
        i_output["psar_reversal"] = psar[:, 3]
