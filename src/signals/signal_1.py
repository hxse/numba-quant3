from numba import njit


from src.utils.constants import numba_config


cache = numba_config["cache"]


@njit(cache=cache)
def calc_signal_1(signal_item, indicator_item, close):
    bbands_upper = indicator_item["bbands_upper"]
    bbands_middle = indicator_item["bbands_middle"]
    bbands_lower = indicator_item["bbands_lower"]
    signal_item["enter_long"] = close < bbands_lower
    signal_item["exit_long"] = close > bbands_middle
    signal_item["enter_short"] = close > bbands_upper
    signal_item["exit_short"] = close < bbands_middle
