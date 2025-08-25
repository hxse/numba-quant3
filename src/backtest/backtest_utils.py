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


cache = numba_config["cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


@njit(cache=cache)
def process_trade_logic(
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


@njit(cache=cache)
def calc_balance(
    i,
    last_i,
    open_arr,
    close_arr,
    position,
    entry_price,
    exit_price,
    equity,
    balance,
    drawdown,
    max_balance,
):
    """
    计算平衡、净值和回撤。
    """
    balance[i] = balance[last_i]
    equity[i] = balance[i]

    # 平仓和反手，更新 balance
    if (
        position[i] == ps.EXIT_LONG.value or position[i] == ps.REVERSE_TO_SHORT.value
    ) and is_long_position(position[last_i]):
        profit_pct = (exit_price[i] - entry_price[last_i]) / entry_price[last_i]
        balance[i] = balance[last_i] * (1 + profit_pct)
        equity[i] = balance[i]
    elif (
        position[i] == ps.EXIT_SHORT.value or position[i] == ps.REVERSE_TO_LONG.value
    ) and is_short_position(position[last_i]):
        profit_pct = (entry_price[last_i] - exit_price[i]) / entry_price[last_i]
        balance[i] = balance[last_i] * (1 + profit_pct)
        equity[i] = balance[i]

    # 持仓状态，计算浮盈并更新 equity（修改为使用 open_arr[i] 以匹配另一个项目）
    elif position[i] == ps.HOLD_LONG.value:
        profit_pct = (open_arr[i] - entry_price[last_i]) / entry_price[last_i]
        equity[i] = balance[last_i] * (1 + profit_pct)
    elif position[i] == ps.HOLD_SHORT.value:
        profit_pct = (entry_price[last_i] - open_arr[i]) / entry_price[last_i]
        equity[i] = balance[last_i] * (1 + profit_pct)

    max_balance[i] = max(max_balance[last_i], balance[i])
    if max_balance[i] > 0:
        drawdown[i] = (max_balance[i] - balance[i]) / max_balance[i]
    else:
        drawdown[i] = 0.0
