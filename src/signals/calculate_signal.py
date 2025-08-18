import numpy as np
import numba as nb
from numba import njit


from .signal_0 import calc_signal_0, signal_0_keys, signal_0_id
from .signal_1 import calc_signal_1, signal_1_keys, signal_1_id
from .signal_2 import calc_signal_2, signal_2_keys, signal_2_id

from src.utils.constants import numba_config


cache = numba_config["cache"]


nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]

signal_dict = {
    signal_0_id: {"func": calc_signal_0, "keys": signal_0_keys},
    signal_1_id: {"func": calc_signal_1, "keys": signal_1_keys},
    signal_2_id: {"func": calc_signal_2, "keys": signal_2_keys},
}


@njit(cache=cache)
def calc_signal(signal_item, indicator_item, close, backtest_params):
    # 使用 if/elif 结构进行分派
    signal_select = backtest_params["signal_select"]

    if signal_select == signal_0_id:
        calc_signal_0(signal_item, indicator_item, close)
    elif signal_select == signal_1_id:
        calc_signal_1(signal_item, indicator_item, close)
    elif signal_select == signal_2_id:
        calc_signal_2(signal_item, indicator_item, close)
