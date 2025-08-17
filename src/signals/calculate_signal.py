from numba import njit


from .signal_0 import calc_signal_0
from .signal_1 import calc_signal_1


from src.utils.constants import numba_config


cache = numba_config["cache"]


@njit(cache=cache)
def calc_signal(signal_item, indicator_item, close, backtest_params):
    # 使用 if/elif 结构进行分派
    if backtest_params["signal_select"] == 0.0:
        calc_signal_0(signal_item, indicator_item, close)
    elif backtest_params["signal_select"] == 1.0:
        calc_signal_1(signal_item, indicator_item, close)
