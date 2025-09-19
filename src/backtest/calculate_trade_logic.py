# backtest_utils.py
import numpy as np
from numba import njit
from numba.core import types

from src.utils.constants import numba_config
from src.backtest.backtest_enums import (
    PositionStatus as ps,
    is_long_position,
    is_short_position,
    is_no_position,
)


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


@njit(cache=enable_cache)
def calc_trade_logic(
    i,
    enter_long_signal,
    exit_long_signal,
    enter_short_signal,
    exit_short_signal,
    position,
    entry_price,
    exit_price,
    target_price,
):
    """
    处理交易逻辑：根据前一根K线的信号和当前仓位状态，更新当前K线的仓位状态和触发价格。
    """
    last_i = i - 1

    # 仓位状态继承
    if is_long_position(position[last_i]):
        position[i] = ps.HOLD_LONG.value
        entry_price[i] = entry_price[last_i]
    elif is_short_position(position[last_i]):
        position[i] = ps.HOLD_SHORT.value
        entry_price[i] = entry_price[last_i]
    else:
        position[i] = ps.NO_POSITION.value
        entry_price[i] = np.nan
        exit_price[i] = np.nan

    # 根据信号处理开平仓逻辑 (优先级：反手 > 平仓 > 开仓)
    if (
        enter_long_signal[last_i]
        and exit_short_signal[last_i]
        and is_short_position(position[last_i])
    ):
        position[i] = ps.REVERSE_TO_LONG.value  # 反手
        entry_price[i] = target_price
        exit_price[i] = target_price
    elif (
        enter_short_signal[last_i]
        and exit_long_signal[last_i]
        and is_long_position(position[last_i])
    ):
        position[i] = ps.REVERSE_TO_SHORT.value  # 反手
        entry_price[i] = target_price
        exit_price[i] = target_price
    elif exit_long_signal[last_i] and is_long_position(position[last_i]):
        position[i] = ps.EXIT_LONG.value  # 平仓
        exit_price[i] = target_price
    elif exit_short_signal[last_i] and is_short_position(position[last_i]):
        position[i] = ps.EXIT_SHORT.value  # 平仓
        exit_price[i] = target_price
    elif enter_long_signal[last_i] and position[last_i] == ps.NO_POSITION.value:
        position[i] = ps.ENTER_LONG.value  # 开多
        entry_price[i] = target_price
    elif enter_short_signal[last_i] and position[last_i] == ps.NO_POSITION.value:
        position[i] = ps.ENTER_SHORT.value  # 开空
        entry_price[i] = target_price
