import numpy as np
from numba import njit, types, literal_unroll
from numba.typed import List, Dict
from numba.core.types import unicode_type
from src.utils.nb_check_keys import check_data_for_signal
from src.convert_params.param_template import get_indicator_need_keys
from src.indicators.calculate_indicators import MaxIndicatorCount as mic


from src.utils.constants import numba_config


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


@njit(cache=enable_cache)
def get_i_output_mtf_need_keys():
    dict_type = Dict.empty(unicode_type, types.float64)
    value_list = List.empty_list(dict_type)
    dict_type2 = Dict.empty(unicode_type, types.float64)
    optim_list = List.empty_list(dict_type2)

    num = 0
    bbands_params = (
        (num, 0, "bbands", "enable", np.array([True, True], dtype=np.float64)),
        (num, 0, "bbands", "period", np.array([14, 5, 50, 1], dtype=np.float64)),
        (num, 0, "bbands", "std_mult", np.array([2, 1, 4, 0.5], dtype=np.float64)),
    )
    assert bbands_params[-1][1] < mic.bbands.value, (
        f"bbands数量超出最大限制 {bbands_params[-1][1]} {mic.bbands.value}"
    )
    num += 1
    sma_params = (
        (num, 0, "sma", "enable", np.array([True, True], dtype=np.float64)),
        (num, 0, "sma", "period", np.array([14, 10, 200, 5], dtype=np.float64)),
        # SMA 1
        (num, 1, "sma", "enable", np.array([True, True], dtype=np.float64)),
        (num, 1, "sma", "period", np.array([200, 100, 40, 10], dtype=np.float64)),
    )
    assert sma_params[-1][1] < mic.sma.value, (
        f"sma数量超出最大限制 {sma_params[-1][1]} {mic.sma.value}"
    )
    all_indicator_data = (*bbands_params, *sma_params)

    for i in range(num + 1):
        value_dict = Dict.empty(unicode_type, types.float64)
        optim_dict = Dict.empty(unicode_type, types.float64)
        value_list.append(value_dict)
        optim_list.append(optim_dict)

    for param_tuple in all_indicator_data:
        _n, _n2, _s, s2, arr = param_tuple

        _value_dict = value_list[_n]
        _optim_dict = optim_list[_n]

        if s2 == "enable":
            if len(arr) >= 1:
                _value_dict[f"{_s}_{s2}_{_n2}"] = arr[0]
            if len(arr) >= 2:
                _optim_dict[f"{_s}_{'optim'}_{_n2}"] = arr[1]
        elif len(arr) == 4:
            _value_dict[f"{_s}_{s2}_{_n2}"] = arr[0]
            _optim_dict[f"{_s}_{s2}_min_{_n2}"] = arr[1]
            _optim_dict[f"{_s}_{s2}_max_{_n2}"] = arr[2]
            _optim_dict[f"{_s}_{s2}_step_{_n2}"] = arr[3]

    return value_list, optim_list


@njit(cache=enable_cache)
def calc_signal_3(
    ohlcv_mtf,
    data_mapping,
    i_output_mtf,
    s_output,
):
    if not check_data_for_signal(
        ohlcv_mtf,
        get_indicator_need_keys(*get_i_output_mtf_need_keys()),
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
