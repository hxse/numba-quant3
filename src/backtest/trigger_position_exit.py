# trigger_position_exit.py
import numpy as np
from numba import njit
from numba.core import types

from src.utils.constants import numba_config
from src.backtest.backtest_enums import PositionStatus as ps
from src.indicators.psar import psar_init, psar_first_iteration, psar_update

from src.backtest.backtest_enums import is_long_position, is_short_position


cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


@njit(cache=cache)
def calculate_exit_triggers(
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
    """
    不知道为什么, 必须得把字典里的参数标量打包成tuple传参才行,如backtest_params_tuple, 200并发4万K线 1.33秒
    反过来如果把数组打包在一起传参, 又会拖慢性能, 200并发4万k线 2.13秒
    """
    last_i = i - 1

    (
        close_for_reversal,
        pct_sl_enable,
        pct_tp_enable,
        pct_tsl_enable,
        pct_sl,
        pct_tp,
        pct_tsl,
        atr_sl_enable,
        atr_tp_enable,
        atr_tsl_enable,
        atr_sl_multiplier,
        atr_tp_multiplier,
        atr_tsl_multiplier,
        psar_enable,
        psar_af0,
        psar_af_step,
        psar_max_af,
    ) = backtest_params_tuple

    target_price = open_arr[i]

    atr = atr_arr[i]
    atr_sl = atr * atr_sl_multiplier
    atr_tp = atr * atr_tp_multiplier
    atr_tsl = atr * atr_tsl_multiplier

    high_prev = high_arr[last_i]
    high_curr = high_arr[i]
    low_prev = low_arr[last_i]
    low_curr = low_arr[i]
    close_prev = close_arr[last_i]

    # 开仓或反手时，初始化止损价格和 PSAR
    if position[i] == ps.ENTER_LONG.value or position[i] == ps.REVERSE_TO_LONG.value:
        pct_sl_arr[i] = target_price * (1 - pct_sl)
        pct_tp_arr[i] = target_price * (1 + pct_tp)
        pct_tsl_arr[i] = target_price * (1 - pct_tsl)
        atr_sl_arr[i] = target_price - atr_sl
        atr_tp_arr[i] = target_price + atr_tp
        atr_tsl_arr[i] = target_price - atr_tsl

        initial_state = psar_init(
            high_prev,
            high_curr,
            low_prev,
            low_curr,
            close_prev,
            1,  # 强制多头
            psar_af0,
        )
        psar_is_long_arr[i], psar_current_arr[i], psar_ep_arr[i], psar_af_arr[i] = (
            initial_state
        )

        (new_state, psar_long, psar_short, reversal) = psar_first_iteration(
            high_prev,
            high_curr,
            low_prev,
            low_curr,
            close_prev,
            psar_af0,
            psar_af_step,
            psar_max_af,
        )
        psar_is_long_arr[i], psar_current_arr[i], psar_ep_arr[i], psar_af_arr[i] = (
            new_state
        )
        psar_reversal_arr[i] = reversal

    elif (
        position[i] == ps.ENTER_SHORT.value or position[i] == ps.REVERSE_TO_SHORT.value
    ):
        pct_sl_arr[i] = target_price * (1 + pct_sl)
        pct_tp_arr[i] = target_price * (1 - pct_tp)
        pct_tsl_arr[i] = target_price * (1 + pct_tsl)
        atr_sl_arr[i] = target_price + atr_sl
        atr_tp_arr[i] = target_price - atr_tp
        atr_tsl_arr[i] = target_price + atr_tsl

        initial_state = psar_init(
            high_prev,
            high_curr,
            low_prev,
            low_curr,
            close_prev,
            -1,  # 强制空头
            psar_af0,
        )
        psar_is_long_arr[i], psar_current_arr[i], psar_ep_arr[i], psar_af_arr[i] = (
            initial_state
        )

        (new_state, psar_long, psar_short, reversal) = psar_first_iteration(
            high_prev,
            high_curr,
            low_prev,
            low_curr,
            close_prev,
            psar_af0,
            psar_af_step,
            psar_max_af,
        )
        psar_is_long_arr[i], psar_current_arr[i], psar_ep_arr[i], psar_af_arr[i] = (
            new_state
        )
        psar_reversal_arr[i] = reversal

    # 持仓时，更新 TSL 和 PSAR
    elif position[i] == ps.HOLD_LONG.value:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else low_arr[i]
        pct_sl_arr[i] = pct_sl_arr[last_i]
        pct_tp_arr[i] = pct_tp_arr[last_i]
        pct_tsl_arr[i] = max(pct_tsl_arr[last_i], exit_check_price * (1 - pct_tsl))
        atr_sl_arr[i] = atr_sl_arr[last_i]
        atr_tp_arr[i] = atr_tp_arr[last_i]
        atr_tsl_arr[i] = max(atr_tsl_arr[last_i], exit_check_price - atr_tsl)

        prev_state = (
            psar_is_long_arr[last_i],
            psar_current_arr[last_i],
            psar_ep_arr[last_i],
            psar_af_arr[last_i],
        )
        (new_state, psar_long, psar_short, reversal) = psar_update(
            prev_state,
            high_arr[i],
            low_arr[i],
            high_arr[last_i],
            low_arr[last_i],
            psar_af_step,
            psar_max_af,
        )
        psar_is_long_arr[i], psar_current_arr[i], psar_ep_arr[i], psar_af_arr[i] = (
            new_state
        )
        psar_reversal_arr[i] = reversal

    elif position[i] == ps.HOLD_SHORT.value:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else high_arr[i]
        pct_sl_arr[i] = pct_sl_arr[last_i]
        pct_tp_arr[i] = pct_tp_arr[last_i]
        pct_tsl_arr[i] = min(pct_tsl_arr[last_i], exit_check_price * (1 + pct_tsl))
        atr_sl_arr[i] = atr_sl_arr[last_i]
        atr_tp_arr[i] = atr_tp_arr[last_i]
        atr_tsl_arr[i] = min(atr_tsl_arr[last_i], exit_check_price + atr_tsl)

        prev_state = (
            psar_is_long_arr[last_i],
            psar_current_arr[last_i],
            psar_ep_arr[last_i],
            psar_af_arr[last_i],
        )
        (new_state, psar_long, psar_short, reversal) = psar_update(
            prev_state,
            high_arr[i],
            low_arr[i],
            high_arr[last_i],
            low_arr[last_i],
            psar_af_step,
            psar_max_af,
        )
        psar_is_long_arr[i], psar_current_arr[i], psar_ep_arr[i], psar_af_arr[i] = (
            new_state
        )
        psar_reversal_arr[i] = reversal

    # 生成离场触发信号 (优化点 2: 简化布尔逻辑)
    if is_long_position(position[i]):
        exit_check_price = close_arr[i] if close_for_reversal > 0 else low_arr[i]

        is_exit = (
            (pct_sl_enable > 0 and exit_check_price < pct_sl_arr[i])
            or (pct_tp_enable > 0 and exit_check_price > pct_tp_arr[i])
            or (pct_tsl_enable > 0 and exit_check_price < pct_tsl_arr[i])
            or (atr_sl_enable > 0 and exit_check_price < atr_sl_arr[i])
            or (atr_tp_enable > 0 and exit_check_price > atr_tp_arr[i])
            or (atr_tsl_enable > 0 and exit_check_price < atr_tsl_arr[i])
            or (psar_enable > 0 and psar_reversal_arr[i] == 1.0)
        )

        if is_exit:
            exit_long_signal[i] = True
            enter_long_signal[i] = False
            exit_short_signal[i] = False

    elif is_short_position(position[i]):
        exit_check_price = close_arr[i] if close_for_reversal > 0 else high_arr[i]

        is_exit = (
            (pct_sl_enable > 0 and exit_check_price > pct_sl_arr[i])
            or (pct_tp_enable > 0 and exit_check_price < pct_tp_arr[i])
            or (pct_tsl_enable > 0 and exit_check_price > pct_tsl_arr[i])
            or (atr_sl_enable > 0 and exit_check_price > atr_sl_arr[i])
            or (atr_tp_enable > 0 and exit_check_price < atr_tp_arr[i])
            or (atr_tsl_enable > 0 and exit_check_price > atr_tsl_arr[i])
            or (psar_enable > 0 and psar_reversal_arr[i] == 1.0)
        )

        if is_exit:
            exit_short_signal[i] = True
            enter_short_signal[i] = False
            exit_long_signal[i] = False
