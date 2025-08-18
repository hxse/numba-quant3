import numpy as np
from numba import njit
from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_keys


cache = numba_config["cache"]

performance_keys = ("position", "price", "money")


@njit(cache=cache)
def calc_performance(performance_item, backtest_item, close, backtest_params):
    exist_key = check_keys(performance_keys, backtest_item)
    if not exist_key:
        return

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
