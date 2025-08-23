import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict, List

from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_keys, check_tohlcv_keys


cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]


@njit(cache=cache)
def get_backtest_keys():
    _l = List.empty_list(types.unicode_type)
    for i in ("enter_long", "exit_long", "enter_short", "exit_short"):
        _l.append(i)
    return _l


@njit(cache=cache)
def calc_backtest(tohlcv, backtest_params, signal_output, backtest_output):
    exist_key = check_keys(get_backtest_keys(), signal_output)
    if not exist_key:
        return

    if not check_tohlcv_keys(tohlcv):
        return

    close = tohlcv["close"]
    data_count = len(close)

    position = np.empty(data_count, dtype=nb_float)
    price = np.empty(data_count, dtype=nb_float)
    money = np.empty(data_count, dtype=nb_float)
    enter_long = signal_output["enter_long"]
    exit_long = signal_output["exit_long"]

    for i in range(data_count):
        money[i] = close[i]
        if enter_long[i]:
            position[i] = 1
            price[i] = close[i]
        elif exit_long[i]:
            position[i] = 0
            price[i] = close[i]

    backtest_output["position"] = position
    backtest_output["price"] = price
    backtest_output["money"] = money
