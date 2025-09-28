import numpy as np
from numba import njit, types
from numba.typed import List
from src.convert_params.param_template import get_indicator_need_keys
from src.signals.tool import populate_indicator_dicts
from src.utils.nb_check_keys import check_data_for_signal
from src.indicators.calculate_indicators import MaxIndicatorCount as mic
from parallel_signature import signal_child_signature

from src.utils.constants import numba_config


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


define_signal_1_params = [
    [
        {
            "name": "sma",
            "enable": True,
            "period": [True, 14, 5, 60, 3, None],
        },
        {
            "name": "sma",
            "enable": True,
            "period": [True, 200, 100, 400, 10, None],
        },
    ],
    [],
]


@njit(signal_child_signature, cache=enable_cache)
def calc_signal_1(
    ohlcv_mtf,
    data_mapping,
    i_params_mtf,
    i_output_mtf,
    s_output,
):
    if not check_data_for_signal(
        ohlcv_mtf,
        get_indicator_need_keys(i_params_mtf),
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
