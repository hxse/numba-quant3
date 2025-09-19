import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict, List

from src.convert_params.nb_params_signature import (
    get_indicator_params_signature,
    get_backtest_params_signature,
)

from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_float = numba_config["nb"]["float"]


@njit(get_indicator_params_signature, cache=enable_cache)
def get_indicator_params(empty):
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    if empty:
        return params
    for i in ["sma", "sma2"]:
        params[f"{i}_enable"] = nb_float(0)
        params[f"{i}_period"] = nb_float(14)

    for i in ["ema", "ema2"]:
        params[f"{i}_enable"] = nb_float(0)
        params[f"{i}_period"] = nb_float(14)

    params["bbands_enable"] = nb_float(0)
    params["bbands_period"] = nb_float(14)
    params["bbands_std_mult"] = nb_float(2.0)

    params["rsi_enable"] = nb_float(0)
    params["rsi_period"] = nb_float(14)

    params["atr_enable"] = nb_float(0)
    params["atr_period"] = nb_float(14)

    params["psar_enable"] = nb_float(0)
    params["psar_af0"] = nb_float(0.02)
    params["psar_af_step"] = nb_float(0.02)
    params["psar_max_af"] = nb_float(0.2)
    return params


@njit(get_backtest_params_signature, cache=enable_cache)
def get_backtest_params(empty):
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    if empty:
        return params

    params["signal_select"] = nb_float(0)

    params["init_money"] = nb_float(10000.0)
    params["close_for_reversal"] = nb_float(1.0)
    params["pct_sl"] = nb_float(0.0)
    params["pct_tp"] = nb_float(0.0)
    params["pct_tsl"] = nb_float(0.0)
    params["pct_sl_enable"] = nb_float(0.0)  # 使用 0.0 或 1.0 代表 False/True
    params["pct_tp_enable"] = nb_float(0.0)
    params["pct_tsl_enable"] = nb_float(0.0)
    params["atr_period"] = nb_float(14.0)
    params["atr_sl_multiplier"] = nb_float(2.0)
    params["atr_tp_multiplier"] = nb_float(2.0)
    params["atr_tsl_multiplier"] = nb_float(2.0)
    params["atr_sl_enable"] = nb_float(0.0)
    params["atr_tp_enable"] = nb_float(0.0)
    params["atr_tsl_enable"] = nb_float(0.0)
    params["psar_enable"] = nb_float(0.0)
    params["psar_af0"] = nb_float(0.02)
    params["psar_af_step"] = nb_float(0.02)
    params["psar_max_af"] = nb_float(0.2)
    params["commission_pct"] = nb_float(0.0)
    params["commission_fixed"] = nb_float(0.0)
    params["slippage_atr"] = nb_float(0.0)
    params["slippage_pct"] = nb_float(0.0)
    params["position_size"] = nb_float(1.0)
    params["annualization_factor"] = nb_float(0.0)

    return params
