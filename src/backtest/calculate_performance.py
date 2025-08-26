import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict, List
from src.utils.constants import numba_config

from src.parallel_signature import performance_signature

from src.backtest.performance_utils import calc_sharpe, calc_calmar, calc_sortino

from src.backtest.backtest_enums import (
    PositionStatus as ps,
    is_long_position,
    is_short_position,
    is_no_position,
)

from src.utils.nb_check_keys import check_keys, check_tohlcv_keys

cache = numba_config["cache"]

nb_float = numba_config["nb"]["float"]


@njit(cache=cache)
def get_performance_keys():
    _l = List.empty_list(types.unicode_type)
    for i in ("position", "entry_price", "exit_price", "equity", "balance", "drawdown"):
        _l.append(i)
    return _l


@njit(cache=cache)
def get_backtest_params_keys():
    _l = List.empty_list(types.unicode_type)
    for i in ("annualization_factor",):
        _l.append(i)
    return _l


@njit(performance_signature, cache=cache)
def calc_performance(tohlcv, backtest_params, backtest_output, performance_output):
    if not check_keys(get_performance_keys(), backtest_output):
        return
    if not check_keys(get_backtest_params_keys(), backtest_params):
        return
    if not check_tohlcv_keys(tohlcv):
        return

    position = backtest_output["position"]
    entry_price = backtest_output["entry_price"]
    exit_price = backtest_output["exit_price"]
    equity = backtest_output["equity"]
    balance = backtest_output["balance"]
    drawdown = backtest_output["drawdown"]

    # ------------------ 计算胜率和盈亏比 ------------------
    # 创建一个列表来存储每笔交易的百分比利润
    profits = []

    # 遍历position数组，找出所有平仓的K线
    for i in range(1, len(position)):
        # 情况1: 平多或反手平多
        if (
            position[i] == ps.EXIT_LONG.value
            or position[i] == ps.REVERSE_TO_SHORT.value
        ) and is_long_position(position[i - 1]):
            profit_pct = (exit_price[i] - entry_price[i - 1]) / entry_price[i - 1]
            profits.append(profit_pct)

        # 情况2: 平空或反手平空
        elif (
            position[i] == ps.EXIT_SHORT.value
            or position[i] == ps.REVERSE_TO_LONG.value
        ) and is_short_position(position[i - 1]):
            profit_pct = (entry_price[i - 1] - exit_price[i]) / entry_price[i - 1]
            profits.append(profit_pct)

    profits_arr = np.array(profits)

    # 胜率
    if len(profits_arr) > 0:
        win_trades = np.sum(profits_arr > 0)
        total_trades = len(profits_arr)
        win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    else:
        win_rate = 0.0

    # 盈亏比
    winning_profits = profits_arr[profits_arr > 0]
    losing_profits = profits_arr[profits_arr < 0]
    if len(winning_profits) > 0 and len(losing_profits) > 0:
        avg_win = np.mean(winning_profits)
        avg_loss = np.mean(losing_profits)
        profit_loss_ratio = avg_win / abs(avg_loss)
    else:
        profit_loss_ratio = 0.0

    # ------------------ 计算最长无仓位时间 ------------------
    longest_no_pos = 0
    current_no_pos = 0
    for i in range(len(position)):
        if is_no_position(position[i]):
            current_no_pos += 1
        else:
            if current_no_pos > longest_no_pos:
                longest_no_pos = current_no_pos
            current_no_pos = 0

    # 循环结束后，再次检查最后一个无仓位序列
    if current_no_pos > longest_no_pos:
        longest_no_pos = current_no_pos

    performance_output["longest_no_position"] = longest_no_pos
    performance_output["win_rate"] = win_rate
    performance_output["profit_loss_ratio"] = profit_loss_ratio

    # ------------------ 原始性能指标 ------------------
    # 定义年化因子，根据K线周期调整
    # 假设您的K线是5分钟，每年有252个交易日，每天8小时交易
    # annualization_factor = 252 * 8 * 12  # 12根5分钟K线/小时
    annualization_factor = backtest_params["annualization_factor"]

    # 计算夏普比率并保存
    sharpe_ratio = calc_sharpe(equity, annualization_factor)
    performance_output["sharpe_ratio"] = sharpe_ratio

    # 计算卡尔马比率并保存
    calmar_ratio = calc_calmar(equity, drawdown, annualization_factor)
    performance_output["calmar_ratio"] = calmar_ratio

    sortino_ratio = calc_sortino(equity, annualization_factor, nb_float(0.0))
    performance_output["sortino_ratio"] = sortino_ratio

    total_profit_pct = (equity[-1] / equity[0]) - 1.0
    performance_output["total_profit_pct"] = total_profit_pct
    performance_output["max_balance"] = np.max(balance)
    performance_output["max_drawdown"] = np.max(drawdown)
