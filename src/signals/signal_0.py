import numpy as np
from numba import njit, types
from numba.typed import List
from src.convert_params.param_template import get_indicator_need_keys
from src.signals.tool import populate_indicator_dicts
from src.utils.nb_check_keys import check_data_for_signal
from src.indicators.calculate_indicators import MaxIndicatorCount as mic

from src.utils.constants import numba_config


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


@njit(cache=enable_cache)
def define_signal_0_params():
    num = 0
    _p = ((num, 0, "", "", np.array([0], dtype=nb_float)),)

    all_indicator_data = (*_p,)

    value_list, optim_list = populate_indicator_dicts(num, all_indicator_data)
    return value_list, optim_list


@njit(cache=enable_cache)
def calc_signal_0(
    ohlcv_mtf,
    data_mapping,
    i_output_mtf,
    s_output,
):
    if not check_data_for_signal(
        ohlcv_mtf,
        get_indicator_need_keys(*define_signal_0_params()),
        i_output_mtf,
        data_mapping,
    ):
        return

    ohlcv_a = ohlcv_mtf[0]
    close = ohlcv_a["close"]
    open = ohlcv_a["open"]

    s_output["enter_long"] = close > open
    s_output["exit_long"] = close < open
    s_output["enter_short"] = close > open
    s_output["exit_short"] = close < open

    skip = data_mapping["skip"]
    s_output["enter_long"][skip == 0] = False
    s_output["enter_short"][skip == 0] = False
