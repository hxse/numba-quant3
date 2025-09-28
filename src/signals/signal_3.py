import numpy as np
from numba import njit, types, literal_unroll
from numba.typed import List, Dict
from numba.core.types import unicode_type
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


define_signal_3_params = [
    [
        {
            "name": "bbands",
            "enable": True,
            # enable_optim, default_value, min_value, max_value, step, current_value
            "period": [True, 14, 5, 50, 1, None],
            "std_mult": [True, 2, 1, 5, 0.5, None],
        }
    ],
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
]


@njit(signal_child_signature, cache=enable_cache)
def calc_signal_3(
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
    i_output_b = i_output_mtf[1]

    bbands_upper = i_output_a["bbands_upper_0"]
    bbands_middle = i_output_a["bbands_middle_0"]
    bbands_lower = i_output_a["bbands_lower_0"]

    mtf_b = data_mapping["mtf_1"]

    sma_0 = i_output_b["sma_0"][mtf_b]
    sma_1 = i_output_b["sma_1"][mtf_b]

    s_output["enter_long"] = (close < bbands_lower) & (sma_0 > bbands_middle)
    s_output["exit_long"] = close > bbands_middle
    s_output["enter_short"] = (close > bbands_upper) & (sma_0 < bbands_middle)
    s_output["exit_short"] = close < bbands_middle

    skip = data_mapping["skip"]
    s_output["enter_long"][skip == 0] = False
    s_output["enter_short"][skip == 0] = False
