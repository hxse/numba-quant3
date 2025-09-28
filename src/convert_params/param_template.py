import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict, List
from src.convert_params.nb_params_signature import (
    get_indicator_params_signature,
    get_backtest_params_signature,
    get_indicator_need_keys_signature,
)
from src.parallel_signature import params_list_type
from src.indicators.calculate_indicators import MaxIndicatorCount as mic
from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]
nb_float = numba_config["nb"]["float"]


@njit(get_indicator_params_signature, cache=enable_cache)
def get_indicator_params(use_presets_indicator_params):
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    if not use_presets_indicator_params:
        return params

    for i in range(mic.sma.value):
        params[f"sma_enable_{i}"] = nb_float(0)
        params[f"sma_period_{i}"] = nb_float(14)

    for i in range(mic.ema.value):
        params[f"ema_enable_{i}"] = nb_float(0)
        params[f"ema_period_{i}"] = nb_float(14)

    for i in range(mic.bbands.value):
        params[f"bbands_enable_{i}"] = nb_float(0)
        params[f"bbands_period_{i}"] = nb_float(14)
        params[f"bbands_std_mult_{i}"] = nb_float(2.0)

    for i in range(mic.rsi.value):
        params[f"rsi_enable_{i}"] = nb_float(0)
        params[f"rsi_period_{i}"] = nb_float(14)

    for i in range(mic.atr.value):
        params[f"atr_enable_{i}"] = nb_float(0)
        params[f"atr_period_{i}"] = nb_float(14)

    for i in range(mic.psar.value):
        params[f"psar_enable_{i}"] = nb_float(0)
        params[f"psar_af0_{i}"] = nb_float(0.02)
        params[f"psar_af_step_{i}"] = nb_float(0.02)
        params[f"psar_max_af_{i}"] = nb_float(0.2)

    return params


@njit(get_indicator_need_keys_signature, cache=enable_cache)
def get_indicator_need_keys(value_list):
    outer_list = List.empty_list(List.empty_list(types.unicode_type))

    for value_dict in value_list:
        inner_list = List.empty_list(types.unicode_type)
        for i in range(mic.sma.value):
            if f"sma_enable_{i}" in value_dict and value_dict[f"sma_enable_{i}"]:
                for v in (f"sma_{i}",):
                    inner_list.append(v)

        for i in range(mic.ema.value):
            if f"ema_enable_{i}" in value_dict and value_dict[f"ema_enable_{i}"]:
                for v in (f"ema_{i}",):
                    inner_list.append(v)

        for i in range(mic.bbands.value):
            if f"bbands_enable_{i}" in value_dict and value_dict[f"bbands_enable_{i}"]:
                for v in (
                    f"bbands_upper_{i}",
                    f"bbands_middle_{i}",
                    f"bbands_lower_{i}",
                ):
                    inner_list.append(v)

        for i in range(mic.rsi.value):
            if f"rsi_enable_{i}" in value_dict and value_dict[f"rsi_enable_{i}"]:
                for v in (f"rsi_{i}",):
                    inner_list.append(v)

        for i in range(mic.atr.value):
            if f"atr_enable_{i}" in value_dict and value_dict[f"atr_enable_{i}"]:
                for v in (f"atr_{i}",):
                    inner_list.append(v)

        for i in range(mic.psar.value):
            if f"psar_enable_{i}" in value_dict and value_dict[f"psar_enable_{i}"]:
                for v in (
                    f"psar_long_{i}",
                    f"psar_short_{i}",
                    f"psar_af_{i}",
                    f"psar_reversal_{i}",
                ):
                    inner_list.append(v)
        outer_list.append(inner_list)
    return outer_list


@njit(get_backtest_params_signature, cache=enable_cache)
def get_backtest_params(use_presets_backtest_params):
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    if not use_presets_backtest_params:
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
