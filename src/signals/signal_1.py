from numba import njit, types
from numba.typed import List


from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_keys


cache = numba_config["cache"]

signal_1_id = 1
signal_1_keys = ("sma", "sma2")


@njit(cache=cache)
def calc_signal_1(signal_item, indicator_item, close):
    exist_key = check_keys(signal_1_keys, indicator_item)
    if not exist_key:
        return

    sma = indicator_item["sma"]
    sma2 = indicator_item["sma2"]
    signal_item["enter_long"] = sma > sma2
    signal_item["exit_long"] = sma < sma2
    signal_item["enter_short"] = sma < sma2
    signal_item["exit_short"] = sma > sma2
