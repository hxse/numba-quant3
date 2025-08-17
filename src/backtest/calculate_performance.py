import numpy as np
from numba import njit
from src.utils.constants import numba_config


cache = numba_config["cache"]


@njit(cache=cache)
def calc_performance(performance_item, backtest_item, close, backtest_params):
    # 这里实现最大回撤、夏普比率等计算
    position = backtest_item["position"]
    price = backtest_item["price"]
    money = backtest_item["money"]

    max_money = 0
    max_drawdown = 0
    for i in range(len(money)):
        max_money = max(money[i], max_money)
        max_drawdown = max(max_money - money[i], max_drawdown)

    performance_item["max_money"] = max_money
    performance_item["max_drawdown"] = max_drawdown
