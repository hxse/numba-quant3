from numba import njit, types
from numba.typed import List


from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_keys

cache = numba_config["cache"]

signal_2_id = 2
signal_2_keys = ("bbands_upper", "bbands_middle", "bbands_lower")


@njit(cache=cache)
def calc_signal_2(signal_item, indicator_item, close):
    exist_key = check_keys(signal_2_keys, indicator_item)
    if not exist_key:
        return

    if exist_key:
        bbands_upper = indicator_item["bbands_upper"]
        bbands_middle = indicator_item["bbands_middle"]
        bbands_lower = indicator_item["bbands_lower"]
        signal_item["enter_long"] = close < bbands_lower
        signal_item["exit_long"] = close > bbands_middle
        signal_item["enter_short"] = close > bbands_upper
        signal_item["exit_short"] = close < bbands_middle
