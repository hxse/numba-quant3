from numba import njit


from src.utils.constants import numba_config


cache = numba_config["cache"]


@njit(cache=cache)
def calc_signal_0(signal_item, indicator_item, close):
    sma = indicator_item["sma"]
    sma2 = indicator_item["sma2"]
    signal_item["enter_long"] = sma > sma2
    signal_item["exit_long"] = sma < sma2
    signal_item["enter_short"] = sma < sma2
    signal_item["exit_short"] = sma > sma2
