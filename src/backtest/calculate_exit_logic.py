# trigger_position_exit.py
import numpy as np
from numba import njit
from numba.core import types

from src.utils.constants import numba_config

from src.backtest.backtest_enums import is_long_position, is_short_position

from backtest.update_exit_targets_utils import update_exit_targets
from backtest.should_trigger_exit_utils import should_trigger_exit

cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


# 修改后: 简洁的主函数
@njit(cache=cache)
def calc_exit_logic(
    i,
    target_price,
    backtest_params_tuple,
    #
    enter_long_signal,
    exit_long_signal,
    enter_short_signal,
    exit_short_signal,
    #
    position,
    entry_price,
    exit_price,
    #
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    #
    atr_arr,
    pct_sl_arr,
    pct_tp_arr,
    pct_tsl_arr,
    atr_sl_arr,
    atr_tp_arr,
    atr_tsl_arr,
    psar_is_long_arr,
    psar_current_arr,
    psar_ep_arr,
    psar_af_arr,
    psar_reversal_arr,
):
    last_i = i - 1

    # 1. 更新所有止损/止盈价格数组
    exit_check_price = update_exit_targets(
        i,
        last_i,
        position,
        target_price,
        backtest_params_tuple,
        high_arr,
        low_arr,
        close_arr,
        atr_arr,
        pct_sl_arr,
        pct_tp_arr,
        pct_tsl_arr,
        atr_sl_arr,
        atr_tp_arr,
        atr_tsl_arr,
        psar_is_long_arr,
        psar_current_arr,
        psar_ep_arr,
        psar_af_arr,
        psar_reversal_arr,
    )

    # 2. 检查离场条件并更新信号
    if is_long_position(position[i]) and should_trigger_exit(
        True,
        exit_check_price,
        psar_reversal_arr[i],
        backtest_params_tuple,
        pct_sl_arr[i],
        pct_tp_arr[i],
        pct_tsl_arr[i],
        atr_sl_arr[i],
        atr_tp_arr[i],
        atr_tsl_arr[i],
    ):
        exit_long_signal[i] = True
        enter_long_signal[i] = False
        exit_short_signal[i] = False

    elif is_short_position(position[i]) and should_trigger_exit(
        False,
        exit_check_price,
        psar_reversal_arr[i],
        backtest_params_tuple,
        pct_sl_arr[i],
        pct_tp_arr[i],
        pct_tsl_arr[i],
        atr_sl_arr[i],
        atr_tp_arr[i],
        atr_tsl_arr[i],
    ):
        exit_short_signal[i] = True
        enter_short_signal[i] = False
        exit_long_signal[i] = False
