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
def define_signal_2_params():
    num = 0
    sma_params = (
        # SMA 0
        (num, 0, "sma", "enable", np.array([True, True], dtype=nb_float)),
        (num, 0, "sma", "period", np.array([14, 6, 200, 5], dtype=nb_float)),
        # SMA 1
        (num, 1, "sma", "enable", np.array([True, True], dtype=nb_float)),
        (num, 1, "sma", "period", np.array([200, 100, 40, 5], dtype=nb_float)),
    )
    assert sma_params[-1][1] < mic.sma.value, (
        f"sma数量超出最大限制 {sma_params[-1][1]} {mic.sma.value}"
    )
    all_indicator_data = (*sma_params,)

    value_list, optim_list = populate_indicator_dicts(num, all_indicator_data)
    return value_list, optim_list


@njit(cache=enable_cache)
def calc_signal_2(
    ohlcv_mtf,
    data_mapping,
    i_output_mtf,
    s_output,
):
    if not check_data_for_signal(
        ohlcv_mtf,
        get_indicator_need_keys(*define_signal_2_params()),
        i_output_mtf,
        data_mapping,
    ):
        return

    ohlcv_a = ohlcv_mtf[0]
    close = ohlcv_a["close"]

    i_output_a = i_output_mtf[0]

    sma_0 = i_output_a["sma_0"]

    sma_1 = i_output_a["sma_1"]

    s_output["enter_long"] = sma_0 > sma_1

    s_output["exit_long"] = sma_0 < sma_1
    s_output["enter_short"] = sma_0 > sma_1
    s_output["exit_short"] = sma_0 < sma_1

    skip = data_mapping["skip"]
    s_output["enter_long"][skip == 0] = False
    s_output["enter_short"][skip == 0] = False
