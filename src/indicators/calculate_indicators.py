import numpy as np
from numba import njit
from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_tohlcv_keys


from .sma import calc_sma
from .bbands import calc_bbands
from .rsi import calc_rsi
from .atr import calc_atr
from .psar import calc_psar

cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]


@njit(cache=cache)
def calc_indicators(tohlcv, indicator_params, indicator_output):
    if not check_tohlcv_keys(tohlcv):
        return

    if "sma_enable" in indicator_params and indicator_params["sma_enable"]:
        indicator_output["sma"] = calc_sma(
            tohlcv["close"], indicator_params["sma_period"]
        )

    if "sma2_enable" in indicator_params and indicator_params["sma2_enable"]:
        indicator_output["sma2"] = calc_sma(
            tohlcv["close"], indicator_params["sma2_period"]
        )

    if "bbands_enable" in indicator_params and indicator_params["bbands_enable"]:
        bbands = calc_bbands(
            tohlcv["close"],
            int(indicator_params["bbands_period"]),
            indicator_params["bbands_std_mult"],
        )
        indicator_output["bbands_upper"] = bbands[:, 0]
        indicator_output["bbands_middle"] = bbands[:, 1]
        indicator_output["bbands_lower"] = bbands[:, 2]
        indicator_output["bbands_bandwidth"] = bbands[:, 3]
        indicator_output["bbands_percent"] = bbands[:, 4]

    if "rsi_enable" in indicator_params and indicator_params["rsi_enable"]:
        indicator_output["rsi"] = calc_rsi(
            tohlcv["close"], indicator_params["rsi_period"]
        )

    if "atr_enable" in indicator_params and indicator_params["atr_enable"]:
        indicator_output["atr"] = calc_atr(
            tohlcv["high"],
            tohlcv["low"],
            tohlcv["close"],
            indicator_params["atr_period"],
        )

    if "psar_enable" in indicator_params and indicator_params["psar_enable"]:
        psar = calc_psar(
            tohlcv["high"],
            tohlcv["low"],
            tohlcv["close"],
            indicator_params["psar_af0"],
            indicator_params["psar_af_step"],
            indicator_params["psar_max_af"],
        )
        indicator_output["psar_long"] = psar[:, 0]
        indicator_output["psar_short"] = psar[:, 1]
        indicator_output["psar_af"] = psar[:, 2]
        indicator_output["psar_reversal"] = psar[:, 3]
