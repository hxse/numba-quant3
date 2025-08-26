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
def update_exit_targets(
    i,
    last_i,
    position,
    target_price,
    backtest_params_tuple,
    #
    high_arr,
    low_arr,
    close_arr,
    atr_arr,
    #
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

    high_prev = high_arr[last_i]
    high_curr = high_arr[i]
    low_prev = low_arr[last_i]
    low_curr = low_arr[i]
    close_prev = close_arr[last_i]
    atr = atr_arr[i]
    atr_sl = atr * atr_sl_multiplier
    atr_tp = atr * atr_tp_multiplier
    atr_tsl = atr * atr_tsl_multiplier
    exit_check_price = 0.0

    # 判断当前是否处于多头仓位或开多
    if is_long_position(position[i]) and position[i] != ps.NO_POSITION.value:
        # 开仓或反手时，初始化价格
        if (
            position[i] == ps.ENTER_LONG.value
            or position[i] == ps.REVERSE_TO_LONG.value
        ):
            pct_sl_arr[i] = target_price * (1 - pct_sl)
            pct_tp_arr[i] = target_price * (1 + pct_tp)
            pct_tsl_arr[i] = target_price * (1 - pct_tsl)
            atr_sl_arr[i] = target_price - atr_sl
            atr_tp_arr[i] = target_price + atr_tp
            atr_tsl_arr[i] = target_price - atr_tsl
            psar_is_long_arr[i], psar_current_arr[i], psar_ep_arr[i], psar_af_arr[i] = (
                psar_init(
                    high_prev, high_curr, low_prev, low_curr, close_prev, 1, psar_af0
                )
            )
            # PSAR第一次迭代需要特殊处理
            (
                (
                    psar_is_long_arr[i],
                    psar_current_arr[i],
                    psar_ep_arr[i],
                    psar_af_arr[i],
                ),
                _,
                _,
                psar_reversal_arr[i],
            ) = psar_first_iteration(
                high_prev,
                high_curr,
                low_prev,
                low_curr,
                close_prev,
                psar_af0,
                psar_af_step,
                psar_max_af,
            )

        # 持仓时，更新跟踪止损和 PSAR
        elif position[i] == ps.HOLD_LONG.value:
            exit_check_price = close_arr[i] if close_for_reversal > 0 else low_arr[i]
            pct_sl_arr[i] = pct_sl_arr[last_i]
            pct_tp_arr[i] = pct_tp_arr[last_i]
            pct_tsl_arr[i] = max(pct_tsl_arr[last_i], exit_check_price * (1 - pct_tsl))
            atr_sl_arr[i] = atr_sl_arr[last_i]
            atr_tp_arr[i] = atr_tp_arr[last_i]
            atr_tsl_arr[i] = max(atr_tsl_arr[last_i], exit_check_price - atr_tsl)

            # 更新 PSAR
            prev_state = (
                psar_is_long_arr[last_i],
                psar_current_arr[last_i],
                psar_ep_arr[last_i],
                psar_af_arr[last_i],
            )
            (
                (
                    psar_is_long_arr[i],
                    psar_current_arr[i],
                    psar_ep_arr[i],
                    psar_af_arr[i],
                ),
                _,
                _,
                psar_reversal_arr[i],
            ) = psar_update(
                prev_state,
                high_arr[i],
                low_arr[i],
                high_arr[last_i],
                low_arr[last_i],
                psar_af_step,
                psar_max_af,
            )

    # 判断当前是否处于空头仓位或开空
    elif is_short_position(position[i]) and position[i] != ps.NO_POSITION.value:
        # 开仓或反手时，初始化价格
        if (
            position[i] == ps.ENTER_SHORT.value
            or position[i] == ps.REVERSE_TO_SHORT.value
        ):
            pct_sl_arr[i] = target_price * (1 + pct_sl)
            pct_tp_arr[i] = target_price * (1 - pct_tp)
            pct_tsl_arr[i] = target_price * (1 + pct_tsl)
            atr_sl_arr[i] = target_price + atr_sl
            atr_tp_arr[i] = target_price - atr_tp
            atr_tsl_arr[i] = target_price + atr_tsl
            psar_is_long_arr[i], psar_current_arr[i], psar_ep_arr[i], psar_af_arr[i] = (
                psar_init(
                    high_prev, high_curr, low_prev, low_curr, close_prev, -1, psar_af0
                )
            )
            # PSAR第一次迭代需要特殊处理
            (
                (
                    psar_is_long_arr[i],
                    psar_current_arr[i],
                    psar_ep_arr[i],
                    psar_af_arr[i],
                ),
                _,
                _,
                psar_reversal_arr[i],
            ) = psar_first_iteration(
                high_prev,
                high_curr,
                low_prev,
                low_curr,
                close_prev,
                psar_af0,
                psar_af_step,
                psar_max_af,
            )

        # 持仓时，更新跟踪止损和 PSAR
        elif position[i] == ps.HOLD_SHORT.value:
            exit_check_price = close_arr[i] if close_for_reversal > 0 else high_arr[i]
            pct_sl_arr[i] = pct_sl_arr[last_i]
            pct_tp_arr[i] = pct_tp_arr[last_i]
            pct_tsl_arr[i] = min(pct_tsl_arr[last_i], exit_check_price * (1 + pct_tsl))
            atr_sl_arr[i] = atr_sl_arr[last_i]
            atr_tp_arr[i] = atr_tp_arr[last_i]
            atr_tsl_arr[i] = min(atr_tsl_arr[last_i], exit_check_price + atr_tsl)

            # 更新 PSAR
            prev_state = (
                psar_is_long_arr[last_i],
                psar_current_arr[last_i],
                psar_ep_arr[last_i],
                psar_af_arr[last_i],
            )
            (
                (
                    psar_is_long_arr[i],
                    psar_current_arr[i],
                    psar_ep_arr[i],
                    psar_af_arr[i],
                ),
                _,
                _,
                psar_reversal_arr[i],
            ) = psar_update(
                prev_state,
                high_arr[i],
                low_arr[i],
                high_arr[last_i],
                low_arr[last_i],
                psar_af_step,
                psar_max_af,
            )
    else:
        # 无仓位时，清空数组
        pct_sl_arr[i] = np.nan
        pct_tp_arr[i] = np.nan
        pct_tsl_arr[i] = np.nan
        atr_sl_arr[i] = np.nan
        atr_tp_arr[i] = np.nan
        atr_tsl_arr[i] = np.nan
        psar_reversal_arr[i] = 0.0
        psar_is_long_arr[i] = 0.0
        psar_current_arr[i] = np.nan
        psar_ep_arr[i] = np.nan
        psar_af_arr[i] = np.nan

    return exit_check_price
