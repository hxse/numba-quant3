import numpy as np
import numba as nb
from numba import njit


from .signal_0 import calc_signal_0, signal_0_keys, signal_0_keys_mtf, signal_0_id
from .signal_1 import calc_signal_1, signal_1_keys, signal_1_keys_mtf, signal_1_id
from .signal_2 import calc_signal_2, signal_2_keys, signal_2_keys_mtf, signal_2_id

from src.utils.constants import numba_config
from src.utils.handle_params import convert_keys

cache = numba_config["cache"]


nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


signal_dict = {
    signal_0_id: {
        "func": calc_signal_0,
        "keys": convert_keys(signal_0_keys),
        "keys_large": convert_keys(signal_0_keys_mtf),
    },
    signal_1_id: {
        "func": calc_signal_1,
        "keys": convert_keys(signal_1_keys),
        "keys_large": convert_keys(signal_1_keys_mtf),
    },
    signal_2_id: {
        "func": calc_signal_2,
        "keys": convert_keys(signal_2_keys),
        "keys_large": convert_keys(signal_2_keys_mtf),
    },
}


@njit(cache=cache)
def calc_signal(
    signal_output,
    indicator_output,
    indicators_output_mtf,
    mapping_mtf,
    close,
    backtest_params,
):
    # 使用 if/elif 结构进行分派
    signal_select = backtest_params["signal_select"]

    if signal_select == signal_0_id:
        calc_signal_0(
            signal_output,
            indicator_output,
            indicators_output_mtf,
            mapping_mtf,
            close,
        )
    elif signal_select == signal_1_id:
        calc_signal_1(
            signal_output,
            indicator_output,
            indicators_output_mtf,
            mapping_mtf,
            close,
        )
    elif signal_select == signal_2_id:
        calc_signal_2(
            signal_output,
            indicator_output,
            indicators_output_mtf,
            mapping_mtf,
            close,
        )
