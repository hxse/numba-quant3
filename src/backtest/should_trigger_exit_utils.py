import numpy as np
from numba import njit
from numba.core import types

from src.utils.constants import numba_config


cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


@njit(cache=cache)
def should_trigger_exit(
    is_long_pos,
    exit_check_price,
    psar_reversal_arr_i,
    backtest_params_tuple,
    #
    pct_sl_arr_i,
    pct_tp_arr_i,
    pct_tsl_arr_i,
    atr_sl_arr_i,
    atr_tp_arr_i,
    atr_tsl_arr_i,
):
    (
        _,
        pct_sl_enable,
        pct_tp_enable,
        pct_tsl_enable,
        _,
        _,
        _,
        atr_sl_enable,
        atr_tp_enable,
        atr_tsl_enable,
        _,
        _,
        _,
        psar_enable,
        _,
        _,
        _,
    ) = backtest_params_tuple

    # 封装所有离场条件
    if is_long_pos:
        return (
            (pct_sl_enable > 0 and exit_check_price < pct_sl_arr_i)
            or (pct_tp_enable > 0 and exit_check_price > pct_tp_arr_i)
            or (pct_tsl_enable > 0 and exit_check_price < pct_tsl_arr_i)
            or (atr_sl_enable > 0 and exit_check_price < atr_sl_arr_i)
            or (atr_tp_enable > 0 and exit_check_price > atr_tp_arr_i)
            or (atr_tsl_enable > 0 and exit_check_price < atr_tsl_arr_i)
            or (psar_enable > 0 and psar_reversal_arr_i == 1.0)
        )
    else:  # 空头
        return (
            (pct_sl_enable > 0 and exit_check_price > pct_sl_arr_i)
            or (pct_tp_enable > 0 and exit_check_price < pct_tp_arr_i)
            or (pct_tsl_enable > 0 and exit_check_price > pct_tsl_arr_i)
            or (atr_sl_enable > 0 and exit_check_price > atr_sl_arr_i)
            or (atr_tp_enable > 0 and exit_check_price < atr_tp_arr_i)
            or (atr_tsl_enable > 0 and exit_check_price > atr_tsl_arr_i)
            or (psar_enable > 0 and psar_reversal_arr_i == 1.0)
        )
