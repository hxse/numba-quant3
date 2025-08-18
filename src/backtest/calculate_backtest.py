import numpy as np
from numba import njit
from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_keys


cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]


backtest_keys = ("enter_long", "exit_long", "enter_short", "exit_short")


@njit(cache=cache)
def calc_backtest(backtest_item, signal_item, close, backtest_params):
    exist_key = check_keys(backtest_keys, signal_item)
    if not exist_key:
        return

    num_data = len(close)
    position = np.empty(num_data, dtype=nb_float)
    price = np.empty(num_data, dtype=nb_float)
    money = np.empty(num_data, dtype=nb_float)
    enter_long = signal_item["enter_long"]
    exit_long = signal_item["exit_long"]

    for i in range(len(close)):
        money[i] = close[i]
        if enter_long[i]:
            position[i] = 1
            price[i] = close[i]
        elif exit_long[i]:
            position[i] = 0
            price[i] = close[i]

    backtest_item["position"] = position
    backtest_item["price"] = price
    backtest_item["money"] = money
