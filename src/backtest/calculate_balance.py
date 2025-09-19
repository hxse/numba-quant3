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
def _calculate_costs(
    nominal_capital,
    slippage_atr,
    slippage_pct,
    commission_pct,
    commission_fixed,
    atr_arr_i,
    position_size,
):
    """
    辅助函数：计算并返回滑点和手续费的总成本。
    """
    # 计算滑点成本
    slippage_cost = 0.0
    if slippage_atr > 0:
        slippage_cost += slippage_atr * atr_arr_i * position_size
    if slippage_pct > 0:
        slippage_cost += nominal_capital * slippage_pct

    # 计算手续费成本
    commission_cost = 0.0
    if commission_pct > 0:
        commission_cost += nominal_capital * 2 * commission_pct
    if commission_fixed > 0:
        commission_cost += commission_fixed * 2

    return slippage_cost + commission_cost


@njit(cache=enable_cache)
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
    max_equity,
    atr_arr,
    commission_pct,
    commission_fixed,
    slippage_atr,
    slippage_pct,
    position_size,
):
    """
    计算平衡、净值和回撤。
    """
    # 1. 首先，账户余额和净值从上一根K线继承
    balance[i] = balance[last_i]
    equity[i] = equity[last_i]

    # 将名义本金的计算提取到函数最上方
    nominal_capital = balance[last_i] * position_size

    if position_size <= 0:
        return

    # 2. 如果发生平仓/反手，更新 balance 和 equity
    if (
        position[i] == ps.EXIT_LONG.value or position[i] == ps.REVERSE_TO_SHORT.value
    ) and is_long_position(position[last_i]):
        profit_pct = (exit_price[i] - entry_price[last_i]) / entry_price[last_i]

        # 调用辅助函数计算总成本
        total_cost = _calculate_costs(
            nominal_capital,
            slippage_atr,
            slippage_pct,
            commission_pct,
            commission_fixed,
            atr_arr[i],
            position_size,
        )

        nominal_profit = nominal_capital * profit_pct
        balance[i] = balance[last_i] + nominal_profit - total_cost
        equity[i] = balance[i]

    elif (
        position[i] == ps.EXIT_SHORT.value or position[i] == ps.REVERSE_TO_LONG.value
    ) and is_short_position(position[last_i]):
        profit_pct = (entry_price[last_i] - exit_price[i]) / entry_price[last_i]

        # 调用辅助函数计算总成本
        total_cost = _calculate_costs(
            nominal_capital,
            slippage_atr,
            slippage_pct,
            commission_pct,
            commission_fixed,
            atr_arr[i],
            position_size,
        )

        nominal_profit = nominal_capital * profit_pct
        balance[i] = balance[last_i] + nominal_profit - total_cost
        equity[i] = balance[i]

    # 3. 持仓时，计算浮动盈亏并更新 equity
    elif is_long_position(position[i]):
        profit_pct = (close_arr[i] - entry_price[i]) / entry_price[i]
        nominal_profit = nominal_capital * profit_pct
        equity[i] = balance[last_i] + nominal_profit

    elif is_short_position(position[i]):
        profit_pct = (entry_price[i] - close_arr[i]) / entry_price[i]
        nominal_profit = nominal_capital * profit_pct
        equity[i] = balance[last_i] + nominal_profit

    # 4. 更新最大净值和回撤
    max_equity[i] = max(max_equity[last_i], equity[i])
    if max_equity[i] > 0:
        drawdown[i] = (max_equity[i] - equity[i]) / max_equity[i]
    else:
        drawdown[i] = 0.0
