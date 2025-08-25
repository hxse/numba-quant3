import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict, List
from src.utils.constants import numba_config

from src.parallel_signature import performance_signature

from src.utils.nb_check_keys import check_keys, check_tohlcv_keys


cache = numba_config["cache"]


@njit(cache=cache)
def get_performance_keys():
    _l = List.empty_list(types.unicode_type)
    for i in ("position", "price", "money"):
        _l.append(i)
    return _l


@njit(performance_signature, cache=cache)
def calc_performance(tohlcv, backtest_params, backtest_output, performance_output):
    exist_key = check_keys(get_performance_keys(), backtest_output)
    if not exist_key:
        return

    if not check_tohlcv_keys(tohlcv):
        return

    position = backtest_output["position"]
    price = backtest_output["price"]
    money = backtest_output["money"]

    max_money = 0
    max_drawdown = 0
    for i in range(len(money)):
        max_money = max(money[i], max_money)
        max_drawdown = max(max_money - money[i], max_drawdown)

    performance_output["max_money"] = max_money
    performance_output["max_drawdown"] = max_drawdown
